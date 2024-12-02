import os
import sys
import time
import json
import math
import random
import datetime
import argparse
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, WeightedRandomSampler

import pickle
import numpy as np
from sklearn.metrics import f1_score

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import train
import utils
from train import *
from logger import Logger
from losses import get_loss
from args import args_parse
from transform import transform
from models.model import get_model
from reweighting import WeightedLoss
from dataset.dataset import get_dataset


global logger
logger = None
sns.set(context="paper", style="white")


if __name__ == "__main__":
    args = args_parse()

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['NCCL_DEBUG']='INFO'
    # os.environ['TORCH_DISTRIBUTED_DETAIL'] = 'DEBUG'
    # os.environ['NCCL_P2P_DISABLE'] = '1'

    # multi-gpu compatibility
    utils.init_distributed_mode(args)
    if utils.is_main_process():
        # concat stat result save path and model save path and mkdir
        timestamp = time.strftime("%m-%d-%H:%M:%S", time.localtime(time.time()))
        args.run_name = f"{args.run_name}{args.arch}_{args.model_selection_split}_{timestamp}_lr_{args.lr}_batch_size_{args.batch_size}_seed{args.seed}_{args.dataset}"
        if args.mixup:
            args.run_name = args.run_name + "_mixup"
        if args.fine_tuning:
            args.run_name = args.run_name + "_fine_tuning_" + args.fine_tuning
        elif args.resume:
            args.run_name = args.run_name + "_resume"
        result_dir = os.path.join(args.result_dir, args.run_name)
        model_save_dir = os.path.join(
            result_dir, "./tmodels" if args.save_each_epoch else "models"
        )
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
    else:
        result_dir = model_save_dir = ""

    logger = Logger(
        logger_name="logger",
        log_file_path=os.path.join(result_dir, "log"),
        enable=utils.get_rank(),
    )
    logger.info(str(args))

    logger.info(f"seed: {args.seed + args.gpu}")
    utils.set_seed(args.seed + args.gpu)

    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        exit(0)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    # device = torch.device(f'cpu')

    args.all_split = [args.train_split] + args.eval_split
    selection_split = args.model_selection_split  # for model selection
    selection_metric = args.model_selection_metric  # for model selection
    if selection_split not in args.eval_split or selection_metric not in args.metric:
        logger.error(
            f"model selection split {selection_split} should in args.eval_split "
            + f"and model selection metric {selection_metric} should in args.metric"
        )
        exit(0)

    # data preparation
    dataset = get_dataset(args.dataset)
    args.dataset_kwargs = dataset.dataset_kwargs

    # augment
    target_transform = None  # transform.target_transform(dataset=dataset)
    train_transform = transform.get_transform(args.train_transform)(
        args.scale, dataset.normMean, dataset.normStd, **args.augment_kwargs
    )
    test_transform = transform.get_transform(args.test_transform)(
        args.scale, dataset.normMean, dataset.normStd, **args.test_augment_kwargs
    )

    # data preparation
    reweighting = "NoneWeighting" if args.method in ["ERM", "MT"] else args.method
    train_dataset = dataset.dataset_split(
        args.root_dir,
        args.train_split,
        train_transform,
        target_transform,
        reweighting,
        args.weights_path,
    )
    eval_splits = {
        split: dataset.dataset_split(
            args.root_dir, split, test_transform, target_transform
        )
        for split in args.eval_split
    }

    if args.resample:
        # class_sample_counts = train_dataset.target_cnt
        # weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
        # samples_weights = weights[train_dataset.targets]
        # resample_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True) # replacement
        pass
    else:
        resample_sampler = None

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if utils.is_dist_avail_and_initialized()
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **args.dataset_kwargs,
    )
    eval_loader = {
        split: DataLoader(
            eval_splits[split],
            batch_size=args.eval_batch_size,
            shuffle=False,
            sampler=None,
            drop_last=False,
            **args.dataset_kwargs,
        )
        for split in args.eval_split
    }
    logger.info(f"train dataset size: {len(train_loader.dataset)}")
    for split, loader in eval_loader.items():
        logger.info(f"{split} dataset size: {len(loader.dataset)}")

    # model preparation
    model = get_model(args, dataset.num_classes, num_points=dataset.num_points)
    if args.model_path:
        logger.info("loading weights at {}".format(args.model_path))
        state = torch.load(args.model_path)
        model.load_state_dict(state["model"])
    else:
        logger.info("using defalut IMAGENET weights")
    model.to(device)
    if utils.is_dist_avail_and_initialized():
        if utils.has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )
    logger.info(
        f"model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if utils.is_main_process():
        stat = defaultdict()
        for metric in args.metric:
            stat[metric] = defaultdict(list)
        stat[selection_metric]["best"] = 0

        logname = os.path.join(result_dir, f"result_{timestamp}.csv")
        with open(logname, "a") as log:
            csv_head = ",".join(
                [
                    f"{split} {metric}"
                    for split in args.all_split
                    for metric in args.metric
                ]
            )
            log.write(csv_head + "\n")

    if args.fine_tuning:
        if args.fine_tuning == "clf":
            # freeze backbone layers
            model.requires_grad_(False)
            model.fc.requires_grad_()
            parameters = model.fc.parameters()
        elif args.fine_tuning == "backbone":
            # freeze clf layers
            model.fc.requires_grad_(False)
            parameters = filter(lambda p: p.requires_grad, model.parameters())
        else:
            parameters = model.parameters()
    else:
        parameters = model.parameters()

    start_epoch = 1

    train_method = (
        args.method if args.method in ["ERM", "DU", "ADU", "MT"] else "REWEIGHTING"
    )
    args.train_method = train_method
    assert hasattr(train, train_method), f"unsupported method: {train_method}"
    train_method = getattr(train, train_method)

    optimizer_name = args.optimizer
    assert hasattr(
        torch.optim, optimizer_name
    ), f"unsupported optimizer name: {optimizer_name}"
    optimizer = getattr(torch.optim, optimizer_name)(
        parameters, lr=args.lr, weight_decay=args.weight_decay, **args.optimizer_kwargs
    )

    criterion_name = args.loss
    if criterion_name != "CrossEntropyLoss":
        args.class_cnt = train_dataset.class_cnt
        args.num_classes = dataset.num_classes
    criterion = get_loss(
        args,
        **args.loss_kwargs,
        reduction="none" if args.train_method == "REWEIGHTING" and args.bw else "mean",
    )
    if args.method in ["DU", "ADU", "MT"]:
        train_criterion = getattr(torch.nn, criterion_name)(reduction="none")
        if args.method == "ADU":
            train_criterion = WeightedLoss(train_criterion, args.weights_path, True)
            train_criterion.to(device)
            # train_dataset.loss_weights = model.module.adu
            optimizer.add_param_group(
                {"params": train_criterion.parameters(), "lr": args.lr}
            )  # , 'weight_decay': 0.0001
        else:
            setattr(train_criterion, "ptw", args.ptw)
    else:
        train_criterion = criterion

    scheduler_name = args.scheduler
    assert hasattr(
        torch.optim.lr_scheduler, scheduler_name
    ), f"unsupported scheduler name: {scheduler_name}"
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
        optimizer, verbose=utils.is_main_process(), **args.scheduler_kwargs
    )

    if args.resume:
        logger.info(f"loading optimizer and scheduler at {args.model_path}")
        start_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    for epoch in range(start_epoch, args.epochs + 1):
        if utils.is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)

        result = train_method(
            dataloader=train_loader,
            model=model,
            criterion=train_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        if utils.is_main_process():
            log = f"epoch: [{epoch}/{args.epochs}]\t "
            for metric in args.metric:
                stat[metric][args.train_split].append(result[metric])
            for metric, value in result.items():
                log += f"{args.train_split} {metric} {value:.3f}\t\t "
            logger.info(log)

            eval_results = {}
            valalidation = {
                "all_val": ["id_val", "val"],
                "robust_val": ["all_val", "aug_val"],
            }
            for split, loader in eval_loader.items():
                if split in valalidation.keys():
                    result = {"y_label": [], "y_predict": [], "loss": [], "len": []}
                    for sub_split in valalidation[split]:
                        result["loss"].append(eval_results[sub_split]["loss"])
                        result["len"].append(len(eval_loader[sub_split].dataset))
                        result["y_label"].extend(eval_results[sub_split]["y_label"])
                        result["y_predict"].extend(eval_results[sub_split]["y_predict"])
                    result["acc"] = sum(
                        np.equal(result["y_label"], result["y_predict"])
                    ) / len(result["y_label"])
                    result["macro_f1"] = f1_score(
                        y_true=result["y_label"],
                        y_pred=result["y_predict"],
                        average="macro",
                    )
                    result["loss"] = sum(
                        np.array(result["len"])
                        / sum(result["len"])
                        * np.array(result["loss"])
                    )
                else:
                    result = eval(
                        dataloader=loader,
                        model=model,
                        criterion=nn.CrossEntropyLoss(),
                        device=device,
                    )
                log = f"epoch: [{epoch}/{args.epochs}]\t "
                for metric in args.metric:
                    log += f"{split} {metric} {result[metric]:.3f}\t\t "
                    stat[metric][split].append(result[metric])
                logger.info(log)
                eval_results[split] = result

            # save model
            model_save_path = f"{model_save_dir}/model_{stat['acc']['test'][-1]:.3f}_f1_{stat['macro_f1']['test'][-1]:.3f}_epo{epoch}_{selection_split}{selection_metric}.pth"
            if (
                args.save_each_epoch
                or stat[selection_metric][selection_split][-1]
                > stat[selection_metric]["best"]
            ):
                # save model
                logger.info(f"saving model at {model_save_path}")
                utils.save_on_master(
                    {
                        "args": args,
                        "epoch": epoch,
                        "arch": args.arch,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    model_save_path,
                )
                if (
                    stat[selection_metric][selection_split][-1]
                    > stat[selection_metric]["best"]
                ):
                    stat[selection_metric]["best"] = stat[selection_metric][
                        selection_split
                    ][-1]
                    logger.info(
                        f'best {selection_metric} at {selection_split} now is {stat["macro_f1"]["test"][-1]:.3f}'
                    )

            # save metric to csv
            with open(logname, "a") as f:
                f.write(
                    (
                        ",".join(["{:.5f}"] * (len(args.all_split) * len(args.metric)))
                        + "\n"
                    ).format(
                        *[
                            stat[metric][split][-1]
                            for split in args.all_split
                            for metric in args.metric
                        ]
                    )
                )

            for metric in args.metric:
                plt.clf()
                sns.lineplot(
                    data={split: stat[metric][split] for split in args.all_split}
                )
                plt.xticks(list(range(epoch)))
                plt.savefig(f"{result_dir}/{metric}.jpg", dpi=300)
                plt.savefig(f"{result_dir}/{metric}.pdf", dpi=300)
                plt.savefig(f"{result_dir}/{metric}.svg", dpi=300)

    pass
