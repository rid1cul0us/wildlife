import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from typing import DefaultDict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import utils
from logger import Logger
from transform import transform
from train import evaluate, evaluateDU
from dataset.dataset import get_dataset
from models.model import get_model, DoubleHeadModel


def args_parse():
    parser = argparse.ArgumentParser(description="ERM")
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        help="model type (default: convnext_tiny)",
    )
    parser.add_argument(
        "--method",
        default="ERM",
        type=str,
        help="determine whether to perform location classification or not",
    )
    parser.add_argument(
        "--batch-size", default=16, type=int, help="batch size (default 32)"
    )
    parser.add_argument(
        "--scale",
        nargs="+",
        type=int,
        default=[448, 448],
        help="size in transforms (default: 224)",
    )

    # model selection & dataset
    parser.add_argument(
        "--dataset",
        default="iwildcam",
        type=str,
        choices=["cct20", "iwildcam", "cct20noempty"],
        help="select dataset the code will use",
    )
    parser.add_argument(
        "--root_dir", default="data/iwildcam", type=str, help="path to dataset root dir"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        nargs="+",
        default=["id_val", "val"],
        help="val and test split names",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["acc", "loss", "macro_f1"],
        help="metrics the algorithm uses",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        default=[
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-22:58:01_lr_4e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.783_f1_0.333_epo10_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-06-02:20:28_lr_4e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.798_f1_0.337_epo9_all_valmacro_f1.pth',
            # 'results/report/bw/PU/resnet50_all_val_08-05_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.1(0.4)_77.8(0.3)_44.6(0.7)_34.9(0.9)/resnet50_all_val_08-05-15:23:09_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.787_f1_0.366_epo14_all_valmacro_f1.pth', # PU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-02:46:42_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.783_f1_0.335_epo17_all_valmacro_f1.pth', # DU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-02:46:42_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.781_f1_0.330_epo16_all_valmacro_f1.pth', # DU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-09:06:58_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.780_f1_0.336_epo12_all_valmacro_f1.pth', # DU
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-19:50:54_lr_4e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.788_f1_0.326_epo5_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-19:50:54_lr_4e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.781_f1_0.341_epo10_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-19:50:54_lr_4e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.795_f1_0.339_epo11_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-19:50:54_lr_4e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.796_f1_0.336_epo14_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-22:58:01_lr_4e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.785_f1_0.329_epo14_all_valmacro_f1.pth', # ERM
            # val macro_f1 select
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-14:36:22_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.791_f1_0.350_epo8_all_valmacro_f1.pth',  # PU
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-17:46:29_lr_5e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.789_f1_0.343_epo9_all_valmacro_f1.pth',  # PU
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-21:07:07_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.798_f1_0.362_epo17_all_valmacro_f1.pth', # PU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-02:46:42_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.754_f1_0.325_epo13_all_valmacro_f1.pth', # DU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-05:55:30_lr_5e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.784_f1_0.315_epo7_all_valmacro_f1.pth',  # DU
            # 'results/report/bw/DU/resnet50_all_val_08-11_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.3(0.6)_78.5(0.4)_45.3(1.7)_32.9(1.1)/resnet50_all_val_08-11-09:06:58_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.791_f1_0.342_epo14_all_valmacro_f1.pth', # DU
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-19:50:54_lr_4e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.795_f1_0.339_epo11_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-05-22:58:01_lr_4e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.793_f1_0.334_epo9_all_valmacro_f1.pth', # ERM
            # 'results/report/ERM/resnet50_all_val_08-05_lr_4e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.6(0.4)_79.2(0.9)_46.9(0.5)_33.3(1.0)/resnet50_all_val_08-06-02:20:28_lr_4e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.798_f1_0.337_epo9_all_valmacro_f1.pth', # ERM
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-14:36:22_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.791_f1_0.350_epo8_all_valmacro_f1.pth', # LoCo
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-17:46:29_lr_5e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.789_f1_0.343_epo9_all_valmacro_f1.pth', # LoCo
            # 'results/report/bw/PU/resnet50_all_val_08-14_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_79.0(0.6)_78.8(1.5)_47.1(0.5)_34.0(0.6)/resnet50_all_val_08-14-21:07:07_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.798_f1_0.362_epo17_all_valmacro_f1.pth', # LoCo
            "results/report/ERM/randaugment/resnet50_all_val_08-23_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.5(0.7)_78.6(1.4)_45.5(1.8)_33.0(1.8)/resnet50_all_val_08-23-13:28:36_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.784_f1_0.310_epo7_all_valmacro_f1.pth",  # ERM
            "results/report/ERM/randaugment/resnet50_all_val_08-23_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.5(0.7)_78.6(1.4)_45.5(1.8)_33.0(1.8)/resnet50_all_val_08-23-16:46:32_lr_5e-05_batch_size_48_seed2_iwildcam/tmodels/model_0.769_f1_0.326_epo10_all_valmacro_f1.pth",  # ERM
            "results/report/ERM/randaugment/resnet50_all_val_08-23_lr_5e-05_batch_size_48_3trials_randaug_op3_iwildcam_78.5(0.7)_78.6(1.4)_45.5(1.8)_33.0(1.8)/resnet50_all_val_08-23-20:29:06_lr_5e-05_batch_size_48_seed3_iwildcam/tmodels/model_0.805_f1_0.354_epo8_all_valmacro_f1.pth",  # ERM
            # 'results/report/bw/DU/ptw_ablation/ptw_0.1/resnet50_all_val_08-26-05:50:50_lr_5e-05_batch_size_48_seed1_iwildcam/tmodels/model_0.776_f1_0.308_epo14_all_valmacro_f1.pth', # CoLoCo
        ],
        help="model path",
    )
    parser.add_argument("--gpu", default=1, type=int, help="gpu to use")
    parser.add_argument(
        "--output_dir",
        # default='analysis/data/best_models',
        type=str,
        help="where to output",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights/iwildcam/global/shifted_softmax_loss_weight_0.4.npy",
        help="losses weights compatible for DU",
    )

    parser.add_argument(
        "--draw_label_probability",
        action="store_true",
        help="whether draw label probability statistics or not",
    )

    parser.add_argument(
        "--y_reorder",
        default="original",
        type=str,
        choices=["reorder", "original"],
        help="whether labels in model output reordered by sample amount, default: original",
    )

    return parser.parse_args()


def draw_cm(
    cm=None, fig_name="fig", title="", xlabels="", ylabels="", value_display=False
):
    sns.heatmap(
        cm,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot=value_display,
    )
    plt.title(title)
    plt.xlabel("true")
    plt.ylabel("predict")
    plt.savefig(f"{fig_name}.jpg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_name}.svg", dpi=300, bbox_inches="tight")
    plt.clf()


def process_confusion_matrix(output_dir, cm=None, cm_path=None):
    if cm_path:
        cm = pickle.load(open(cm_path, "rb"))
        # np.savetxt("cm.csv", cm, delimiter=",")

    cm_percentage = np.zeros((len(cm), len(cm)), np.float32)
    for col in range(len(cm)):
        col_sum = np.sum(cm[:, col])
        for row in range(len(cm)):
            cm_percentage[row][col] = (
                round(cm[row][col] * 1.0 / (col_sum + 1e-8), 2) * 100
            )
    pickle.dump(cm, open(os.path.join(output_dir, "confusion_matrix"), "wb"))
    pickle.dump(
        cm_percentage,
        open(os.path.join(output_dir, "confusion_matrix_percentage"), "wb"),
    )
    draw_cm(
        cm=cm,
        fig_name=os.path.join(output_dir, "cm"),
        xlabels="",
        ylabels="",
        title="confusion matrix",
    )
    draw_cm(
        cm=cm_percentage,
        fig_name=os.path.join(output_dir, "cm_percentage"),
        xlabels="",
        ylabels="",
        title="confusion matrix (%)",
    )


def process_label_probability(probability, output_dir, classnames):
    pickle.dump(probability, open(os.path.join(output_dir, "label_probability"), "wb"))
    probs_output_dir = os.path.join(output_dir, "label_confidence")
    os.makedirs(probs_output_dir, exist_ok=True)
    for ty in ("all", "true", "false"):
        ty_probs_output_dir = os.path.join(probs_output_dir, ty)
        os.makedirs(ty_probs_output_dir, exist_ok=True)

    for name in classnames:
        # fig = sns.displot({'true': probability['true'][name], 'false': probability['false'][name]}, kde=True, )
        # if len(probability['true'][name]) and len(probability['false'][name]):
        if len(probability["true"][name]) == 0 and len(probability["false"][name]) == 0:
            continue
        fig = sns.kdeplot(
            {"true": probability["true"][name], "false": probability["false"][name]}
        )
        fig.set_xlim(0, 1)
        plt.suptitle(
            f'{name} (true:{len(probability["true"][name])} false:{len(probability["false"][name])})'
        )
        plt.savefig(f"{probs_output_dir}/kde_{name}.jpg", dpi=300)
        plt.clf()

        for ty in ("all", "true", "false"):
            # pic = sns.displot(data=probability[ty][name])
            # pic.set_xlabels(f'{ty}')

            fig = sns.histplot(
                probability[ty][name],
                stat="probability",
                binrange=[0, 1.0],
                binwidth=0.1,
            )
            fig.set_ylim(0, 1)
            plt.xlabel("confidence")
            plt.suptitle(f"{name} ({len(probability[ty][name])})")
            plt.savefig(f"{probs_output_dir}/{ty}/{ty}_{name}.jpg", dpi=300)
            plt.clf()


# def transfer(specifics:[str]):
#     for specific in specifics:
#         output_dir = os.path.dirname(specific)
#         filename = os.path.basename(specific).split(':')[0]
#         specific = pickle.load(open(specific, 'rb'))
#         # json : imgname: top5, top5_probs
#         # numpy: probs
#         top5_results = []
#         probs_results = []
#         for item in specific:
#             imgname, y, logits, probs = item
#             top5 = np.argsort(probs)[-5:][::-1].tolist()
#             top5probs = [probs[i] for i in top5]
#             top5_results.append([imgname, y, top5, top5probs])
#             probs_results.append(probs)
#         top5_results_path = os.path.join(output_dir, f'{filename}.json')
#         probs_results_path = os.path.join(output_dir, f'{filename}.npy')
#         json.dump(top5_results, open(top5_results_path, 'w'), indent=4)
#         pickle.dump(probs_results, open(probs_results_path, 'wb'))


# def metric(specifics:[str]):
#     for specific in specifics:
#         top5_specific = json.load(open(specific, 'r')) # [(imgname: top5, top5_probs)]
#         result = np.zeros((182, 182), dtype=np.int32)
#         top5_correct = [0] * 182
#         top5_total = [0] * 182
#         for item in top5_specific:
#             _, y, top5, _ = item
#             # p = top5[0]
#             # result[y][p] += 1
#             top5_correct[y] += 1 if y in top5 else 0
#             # top5_total[y] += 1

#         print(sum(top5_correct) / len(top5_specific))


if __name__ == "__main__":
    args = args_parse()

    for path in args.model_path:
        if os.path.isdir(path):
            best_models = []
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if filename == "best_model.pth":
                        best_models.append(os.path.join(root, filename))
            args.model_path = best_models

    sns.set(context="paper", style="white")
    # plt.style.use('seaborn-colorblind')
    sns.set_palette(sns.color_palette("colorblind"))

    dataset = get_dataset(args.dataset)
    device = torch.device(f"cuda:{args.gpu}")
    models = DefaultDict()
    args.multitask = "DU" in args.method
    models[args.arch] = get_model(args, dataset.num_classes, dataset.num_points)

    # target_transform = None # transform.target_transform(dataset=dataset)
    target_transform = transform.target_transform(dataset=dataset)
    test_transform = transform.test_transform(
        args.scale, dataset.normMean, dataset.normStd
    )

    eval_splits = {
        split: dataset.dataset_split(
            args.root_dir,
            split,
            test_transform,
            target_transform,
            reweighting=args.method
            if split == "id_test" and args.multitask
            else "NoneWeighting",
            weights_path=args.weights_path,
        )
        for split in args.eval_split
    }
    eval_loader = {
        split: DataLoader(
            eval_splits[split],
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            drop_last=False,
            **dataset.dataset_kwargs,
        )
        for split in args.eval_split
    }

    logger = None

    for model_path in args.model_path:
        # set seed
        seed = int(model_path.split("_seed")[1][0])
        utils.set_seed(seed=seed)

        print(f"loading weights at {model_path}")
        state = torch.load(model_path, map_location="cpu")
        if not hasattr(state, "arch"):
            state["arch"] = (
                "convnext_tiny" if "convnext_tiny" in model_path else "resnet50"
            )
        if state["arch"] in models.keys():
            model = models[state["arch"]]
        else:
            model = models[state["arch"]] = get_model(
                state["arch"], dataset.num_classes
            )
        if "module" in list(state["model"].keys())[0]:
            state["model"] = {
                k[7:]: v for k, v in state["model"].items() if "module" in k
            }
        model.load_state_dict(state_dict=state["model"])
        model.to(device)

        eval_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "eval")
        output_dir = os.path.join(eval_dir, os.path.basename(model_path))
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, model_path.split("/")[-1])
            for old, new in {"ERM": "ERM", "PU": "LoCo", "DU": "CoLoCo"}.items():
                if old in model_path:
                    output_dir += "/" + new
        os.makedirs(output_dir, exist_ok=True)

        if logger:
            del logger
        logger = Logger(
            logger_name="logger", log_file_path=os.path.join(output_dir, "log")
        )
        logger.info(f"save at {args.output_dir}")
        logger.info(f"seed {seed}")

        for split, loader in eval_loader.items():
            split_output_dir = os.path.join(output_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            logger.info(f"{split} split size: {len(loader.dataset)}")
            if split == "id_test" and args.multitask:
                results = evaluateDU(
                    dataloader=loader,
                    model=model,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                )
            else:
                results = evaluate(
                    dataloader=loader,
                    model=model,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                )

            # deal with metrics

            logger.info(
                f"{split} loss: {results['loss']} {split} acc: {results['acc']} {split} macro_f1: {results['macro_f1']}"
            )
            if split == "id_test" and args.multitask:
                logger.info(
                    f"{split} p_loss: {results['p_loss']} {split} p_acc: {results['p_acc']} {split} p_macro_f1: {results['p_macro_f1']}"
                )
                print(results["p_str_report"])

            # deal with cm
            if "cm" in results.keys():
                process_confusion_matrix(cm=results["cm"], output_dir=split_output_dir)

            if args.draw_label_probability:
                # deal with label probability
                # id2class = loader.dataset.idx_to_class
                # class2id = loader.dataset.class_to_idx
                classnames = loader.dataset.class_to_idx.keys()
                process_label_probability(
                    probability=results["probs"],
                    output_dir=split_output_dir,
                    classnames=classnames,
                )

            # deal with specific output
            # y reorder
            # if args.y_reorder == 'reorder':
            # logger.info('reorder label order, use y_original to recover')
            # y_original = loader.dataset.y_original
            # results['specific'] = {k: [v[0], y_original[v[1]], y_original[v[2]]] for k, v in results['specific'].items()}
            # json.dump(results['top1_specific'], open(os.path.join(split_output_dir, f'{split}_top1.json'), 'w'), indent=4)
            # json.dump(results['specific'], open(os.path.join(split_output_dir, f'{split}_specific.json'), 'w'), indent=4)

            if "specific" in results.keys():
                specific_list = [
                    [name, y, l, p] for name, (y, l, p) in results["specific"].items()
                ]
                pickle.dump(
                    specific_list,
                    open(os.path.join(split_output_dir, f"{split}_specific"), "wb"),
                )  # save specific
            if "cm" in results.keys():
                results["cm"] = results["cm"].tolist()
            json.dump(
                results,
                open(os.path.join(split_output_dir, f"{split}_results.json"), "w"),
                indent=4,
            )  # save results json
            print(results["str_report"])
