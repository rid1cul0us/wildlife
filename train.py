import os
import json
import random
import numpy as np
import collections
from typing import List, Optional, Union, Tuple, Dict, DefaultDict

import torch
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tenumerate
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import utils
from logger import Logger


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def ERM(dataloader, model, criterion, optimizer, scheduler, device) -> Dict:
    losses = []
    total_cnt = 0
    correct_cnt = 0

    y_label = []
    y_predict = []

    # switch to train mode
    model.train()

    for idx, (img, target, _) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, target = img.to(device), target.to(device)

        output = model(img)

        # measure accuracy and record loss
        loss = criterion(output, target)
        output = torch.nn.functional.softmax(output, dim=1)
        predicts = torch.argmax(output, dim=-1)
        correct = torch.sum(target == predicts)

        y_label.extend(target.int().tolist())
        y_predict.extend(predicts.int().tolist())

        losses.append(loss.item())
        correct_cnt += correct.item()
        total_cnt += target.size(0)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    result_dict = {
        "loss": np.mean(losses),
        "correct_cnt": correct_cnt,
        "total_cnt": total_cnt,
    }
    # assert total_cnt == len(dataloader) * dataloader.batch_size
    result = utils.reduce_dict(input_dict=result_dict, average=False)
    # assert result['total_cnt'] == len(dataloader) * dataloader.batch_size * utils.get_world_size()

    # gather  y_label, y_predict
    y_label, y_predict = (
        torch.tensor(y_label, dtype=torch.int).cuda(),
        torch.tensor(y_predict, dtype=torch.int).cuda(),
    )
    gather_result = utils.gather_dict({"y_label": y_label, "y_predict": y_predict})
    # assert torch.sum(gather_result['y_label'] == gather_result['y_predict']).item() == result['correct_cnt'].item()
    y_label, y_predict = (
        gather_result["y_label"].tolist(),
        gather_result["y_predict"].tolist(),
    )

    macro_f1 = f1_score(y_label, y_predict, average="macro")

    return {
        "macro_f1": macro_f1,
        "loss": result["loss"].item() / utils.get_world_size(),
        "acc": result["correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
    }


def DU(dataloader, model, criterion, optimizer, scheduler, device) -> Dict:
    # loss
    y_losses = []
    p_losses = []
    losses = []

    # correct count
    total_cnt = 0
    y_correct_cnt = 0
    p_correct_cnt = 0

    # labels and predicts
    y_label = []
    y_predict = []
    p_label = []
    p_predict = []

    # switch to train mode
    model.train()

    for idx, (img, target, side_info) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, c_target = img.to(device), target.to(device)
        p_target, c_weight, p_weight = (
            side_info[0].to(device),
            side_info[1].to(device),
            side_info[2].to(device),
        )

        c_output, p_output = model(img)

        # measure accuracy and record loss

        # for class loss
        c_loss = criterion(c_output, c_target)
        c_loss = c_loss * c_weight
        output = torch.nn.functional.softmax(c_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == c_target)

        y_correct_cnt += correct.item()
        y_label.extend(c_target.int().tolist())
        y_predict.extend(predict.int().tolist())

        c_loss = c_loss.mean()
        y_losses.append(c_loss.item())

        # for point loss
        p_loss = criterion(p_output, p_target)
        p_loss = p_loss * p_weight
        output = torch.nn.functional.softmax(p_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == p_target)

        p_correct_cnt += correct.item()
        p_label.extend(p_target.int().tolist())
        p_predict.extend(predict.int().tolist())

        p_loss = p_loss.mean()
        p_losses.append(p_loss.item())

        loss = c_loss + criterion.ptw * p_loss
        losses.append(loss.item())

        total_cnt += c_target.size(0)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    result_dict = {
        "total_loss": np.mean(losses),
        "y_loss": np.mean(y_losses),
        "p_loss": np.mean(p_losses),
        "y_correct_cnt": y_correct_cnt,
        "p_correct_cnt": p_correct_cnt,
        "total_cnt": total_cnt,
    }
    # assert total_cnt == len(dataloader) * dataloader.batch_size
    result = utils.reduce_dict(input_dict=result_dict, average=False)
    # assert result['total_cnt'] == len(dataloader) * dataloader.batch_size * utils.get_world_size()

    # gather  y_label, y_predict
    y_label, y_predict = (
        torch.tensor(y_label, dtype=torch.int).cuda(),
        torch.tensor(y_predict, dtype=torch.int).cuda(),
    )
    p_label, p_predict = (
        torch.tensor(p_label, dtype=torch.int).cuda(),
        torch.tensor(p_predict, dtype=torch.int).cuda(),
    )
    gather_result = utils.gather_dict(
        {
            "y_label": y_label,
            "y_predict": y_predict,
            "p_label": p_label,
            "p_predict": p_predict,
        }
    )
    # assert torch.sum(gather_result['y_label'] == gather_result['y_predict']).item() == result['correct_cnt'].item()
    y_label, y_predict = (
        gather_result["y_label"].tolist(),
        gather_result["y_predict"].tolist(),
    )
    p_label, p_predict = (
        gather_result["p_label"].tolist(),
        gather_result["p_predict"].tolist(),
    )

    y_macro_f1 = f1_score(y_label, y_predict, average="macro")
    p_macro_f1 = f1_score(p_label, p_predict, average="macro")

    return {
        "macro_f1": y_macro_f1,
        "loss": result["y_loss"].item() / utils.get_world_size(),
        "acc": result["y_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "p_macro_f1": p_macro_f1,
        "p_loss": result["p_loss"].item() / utils.get_world_size(),
        "p_acc": result["p_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "total_loss": result["total_loss"].item() / utils.get_world_size(),
    }


def MT(dataloader, model, criterion, optimizer, scheduler, device) -> Dict:
    # loss
    y_losses = []
    p_losses = []
    losses = []

    # correct count
    total_cnt = 0
    y_correct_cnt = 0
    p_correct_cnt = 0

    # labels and predicts
    y_label = []
    y_predict = []
    p_label = []
    p_predict = []

    # switch to train mode
    model.train()

    for idx, (img, target, side_info) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, c_target = img.to(device), target.to(device)
        p_target = side_info[0].to(device)

        c_output, p_output = model(img)

        # measure accuracy and record loss

        # for class loss
        c_loss = criterion(c_output, c_target)
        output = torch.nn.functional.softmax(c_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == c_target)

        y_correct_cnt += correct.item()
        y_label.extend(c_target.int().tolist())
        y_predict.extend(predict.int().tolist())

        c_loss = c_loss.mean()
        y_losses.append(c_loss.item())

        # for point loss
        p_loss = criterion(p_output, p_target)
        output = torch.nn.functional.softmax(p_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == p_target)

        p_correct_cnt += correct.item()
        p_label.extend(p_target.int().tolist())
        p_predict.extend(predict.int().tolist())

        p_loss = p_loss.mean()
        p_losses.append(p_loss.item())

        loss = c_loss + criterion.ptw * p_loss
        losses.append(loss.item())

        total_cnt += c_target.size(0)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    result_dict = {
        "total_loss": np.mean(losses),
        "y_loss": np.mean(y_losses),
        "p_loss": np.mean(p_losses),
        "y_correct_cnt": y_correct_cnt,
        "p_correct_cnt": p_correct_cnt,
        "total_cnt": total_cnt,
    }
    # assert total_cnt == len(dataloader) * dataloader.batch_size
    result = utils.reduce_dict(input_dict=result_dict, average=False)
    # assert result['total_cnt'] == len(dataloader) * dataloader.batch_size * utils.get_world_size()

    # gather  y_label, y_predict
    y_label, y_predict = (
        torch.tensor(y_label, dtype=torch.int).cuda(),
        torch.tensor(y_predict, dtype=torch.int).cuda(),
    )
    p_label, p_predict = (
        torch.tensor(p_label, dtype=torch.int).cuda(),
        torch.tensor(p_predict, dtype=torch.int).cuda(),
    )
    gather_result = utils.gather_dict(
        {
            "y_label": y_label,
            "y_predict": y_predict,
            "p_label": p_label,
            "p_predict": p_predict,
        }
    )
    # assert torch.sum(gather_result['y_label'] == gather_result['y_predict']).item() == result['correct_cnt'].item()
    y_label, y_predict = (
        gather_result["y_label"].tolist(),
        gather_result["y_predict"].tolist(),
    )
    p_label, p_predict = (
        gather_result["p_label"].tolist(),
        gather_result["p_predict"].tolist(),
    )

    y_macro_f1 = f1_score(y_label, y_predict, average="macro")
    p_macro_f1 = f1_score(p_label, p_predict, average="macro")

    return {
        "macro_f1": y_macro_f1,
        "loss": result["y_loss"].item() / utils.get_world_size(),
        "acc": result["y_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "p_macro_f1": p_macro_f1,
        "p_loss": result["p_loss"].item() / utils.get_world_size(),
        "p_acc": result["p_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "total_loss": result["total_loss"].item() / utils.get_world_size(),
    }


def ADU(dataloader, model, criterion, optimizer, scheduler, device) -> Dict:
    # loss
    y_losses = []
    p_losses = []
    losses = []

    # correct count
    total_cnt = 0
    y_correct_cnt = 0
    p_correct_cnt = 0

    # labels and predicts
    y_label = []
    y_predict = []
    p_label = []
    p_predict = []

    # switch to train mode
    model.train()

    for idx, (img, target, side_info) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, c_target = img.to(device), target.to(device)
        p_target, c_index, p_index = (
            side_info[0].to(device),
            side_info[1].to(device),
            side_info[2].to(device),
        )

        c_output, p_output = model(img)

        # measure accuracy and record loss

        # for class loss
        c_loss = criterion(c_output, c_target, c_index, "class")
        output = torch.nn.functional.softmax(c_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == c_target)

        y_correct_cnt += correct.item()
        y_label.extend(c_target.int().tolist())
        y_predict.extend(predict.int().tolist())

        c_loss = c_loss.mean()
        y_losses.append(c_loss.item())

        # for point loss
        p_loss = criterion(p_output, p_target, p_index, "point")
        output = torch.nn.functional.softmax(p_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == p_target)

        p_correct_cnt += correct.item()
        p_label.extend(p_target.int().tolist())
        p_predict.extend(predict.int().tolist())

        p_loss = p_loss.mean()
        p_losses.append(p_loss.item())

        # loss = c_loss + criterion.ptw * p_loss
        loss = c_loss + p_loss
        losses.append(loss.item())

        total_cnt += c_target.size(0)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    result_dict = {
        "total_loss": np.mean(losses),
        "y_loss": np.mean(y_losses),
        "p_loss": np.mean(p_losses),
        "y_correct_cnt": y_correct_cnt,
        "p_correct_cnt": p_correct_cnt,
        "total_cnt": total_cnt,
    }
    # assert total_cnt == len(dataloader) * dataloader.batch_size
    result = utils.reduce_dict(input_dict=result_dict, average=False)
    # assert result['total_cnt'] == len(dataloader) * dataloader.batch_size * utils.get_world_size()

    # gather  y_label, y_predict
    y_label, y_predict = (
        torch.tensor(y_label, dtype=torch.int).cuda(),
        torch.tensor(y_predict, dtype=torch.int).cuda(),
    )
    p_label, p_predict = (
        torch.tensor(p_label, dtype=torch.int).cuda(),
        torch.tensor(p_predict, dtype=torch.int).cuda(),
    )
    gather_result = utils.gather_dict(
        {
            "y_label": y_label,
            "y_predict": y_predict,
            "p_label": p_label,
            "p_predict": p_predict,
        }
    )
    # assert torch.sum(gather_result['y_label'] == gather_result['y_predict']).item() == result['correct_cnt'].item()
    y_label, y_predict = (
        gather_result["y_label"].tolist(),
        gather_result["y_predict"].tolist(),
    )
    p_label, p_predict = (
        gather_result["p_label"].tolist(),
        gather_result["p_predict"].tolist(),
    )

    y_macro_f1 = f1_score(y_label, y_predict, average="macro")
    p_macro_f1 = f1_score(p_label, p_predict, average="macro")

    return {
        "macro_f1": y_macro_f1,
        "loss": result["y_loss"].item() / utils.get_world_size(),
        "acc": result["y_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "p_macro_f1": p_macro_f1,
        "p_loss": result["p_loss"].item() / utils.get_world_size(),
        "p_acc": result["p_correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
        "total_loss": result["total_loss"].item() / utils.get_world_size(),
    }


def REWEIGHTING(dataloader, model, criterion, optimizer, scheduler, device) -> Dict:
    losses = []
    total_cnt = 0
    correct_cnt = 0

    y_label = []
    y_predict = []

    # switch to train mode
    model.train()

    for idx, (img, target, side_info) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, target = img.to(device), target.to(device)
        # weight, _ = side_info[0].to(device), side_info[1].to(device)
        weight = side_info[0].to(device)  # , side_info[1].to(device)

        output = model(img)

        # measure accuracy and record loss

        # for class loss
        loss = criterion(output, target)
        loss = loss * weight
        output = torch.nn.functional.softmax(output, dim=1)
        predicts = torch.argmax(output, dim=-1)
        correct = torch.sum(target == predicts)

        y_label.extend(target.int().tolist())
        y_predict.extend(predicts.int().tolist())

        loss = loss.mean()
        losses.append(loss.item())
        correct_cnt += correct.item()
        total_cnt += target.size(0)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    result_dict = {
        "loss": np.mean(losses),
        "correct_cnt": correct_cnt,
        "total_cnt": total_cnt,
    }
    result = utils.reduce_dict(input_dict=result_dict, average=False)

    # gather  y_label, y_predict
    y_label, y_predict = (
        torch.tensor(y_label, dtype=torch.int).cuda(),
        torch.tensor(y_predict, dtype=torch.int).cuda(),
    )
    gather_result = utils.gather_dict({"y_label": y_label, "y_predict": y_predict})
    y_label, y_predict = (
        gather_result["y_label"].tolist(),
        gather_result["y_predict"].tolist(),
    )

    macro_f1 = f1_score(y_label, y_predict, average="macro")

    return {
        "macro_f1": macro_f1,
        "loss": result["loss"].item() / utils.get_world_size(),
        "acc": result["correct_cnt"].item() * 1.0 / result["total_cnt"].item(),
    }


@torch.no_grad()
def eval(dataloader, model, criterion, device) -> dict:
    losses = []
    total_cnt = 0
    correct_cnt = 0

    y_label = []
    y_predict = []

    # switch to test mode
    model.eval()

    for _, (img, target, _) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, target = img.to(device), target.to(device)

        output = model(img)

        # measure accuracy and record loss
        loss = criterion(output, target)
        output = torch.nn.functional.softmax(output, dim=1)
        predicts = torch.argmax(output, dim=-1)
        correct = torch.sum(target == predicts)

        y_label.extend(target.int().tolist())
        y_predict.extend(predicts.int().tolist())

        losses.append(loss.item())
        correct_cnt += correct.item()
        total_cnt += target.size(0)

    macro_f1 = f1_score(y_label, y_predict, average="macro")

    return {
        "macro_f1": macro_f1,
        "loss": np.mean(losses),
        "acc": correct_cnt * 1.0 / total_cnt,
        "y_label": y_label,
        "y_predict": y_predict,
    }


@torch.no_grad()
def evaluate(dataloader, model, criterion, device, topk=1) -> dict:
    losses = []
    total_cnt = 0
    correct_cnt = 0

    y_label = []
    imgname = []
    y_predict = []

    image_probs = []
    image_logits = []

    id2class = dataloader.dataset.idx_to_class
    class2id = dataloader.dataset.class_to_idx
    num_classes = dataloader.dataset.num_classes
    cm = np.zeros((num_classes, num_classes), np.int32)

    probability = DefaultDict()
    for ty in ("all", "true", "false"):
        probability[ty] = DefaultDict(list)

    # switch to test mode
    model.eval()

    for _, (img, target, (_, imgpath)) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, target = img.to(device), target.to(device)

        logits = model(img)

        # measure accuracy and record loss
        loss = criterion(logits, target)
        probs = torch.nn.functional.softmax(logits, dim=1)
        prob, predicts = torch.max(probs, dim=-1)
        correct = torch.sum(target == predicts)

        losses.append(loss.item())
        correct_cnt += correct.item()
        total_cnt += target.size(0)

        for true, pred, p in zip(target, predicts, prob):
            true, pred, p = true.item(), pred.item(), p.item()
            y_label.append(true)
            y_predict.append(pred)
            cm[pred][true] += 1
            probability["all"][id2class[true]].append(p)
            probability["true" if true == pred else "false"][id2class[true]].append(p)

        image_logits.extend(logits.cpu().tolist())
        image_probs.extend(probs.cpu().tolist())
        imgname.extend(imgpath)

    macro_f1 = f1_score(y_label, y_predict, average="macro")
    top1_specific = {name: [l, p] for name, l, p in zip(imgname, y_label, y_predict)}
    specific = {
        name: [y, l, p]
        for name, y, l, p in zip(imgname, y_label, image_logits, image_probs)
    }

    str_report = classification_report(y_label, y_predict)
    target_names = [id2class[i] for i in range(num_classes)]
    json_report = classification_report(y_label, y_predict, output_dict=True)
    # json_report = {k+'/'+(target_names[int(k)] if '0' <= k[0] <= '9' else ''): v for k, v in json_report.items()}

    json_report.update(
        {
            "cm": cm,
            "specific": specific,
            "top1_specific": top1_specific,
            "macro_f1": macro_f1,
            "probs": probability,
            "loss": np.mean(losses),
            "acc": correct_cnt * 1.0 / total_cnt,
            "str_report": str_report,
            "y_label": y_label,
            "y_predict": y_predict,
        }
    )

    return json_report


@torch.no_grad()
def evaluateDU(dataloader, model, criterion, device) -> Dict:
    # loss
    y_losses = []
    p_losses = []
    losses = []

    # correct count
    total_cnt = 0
    y_correct_cnt = 0
    p_correct_cnt = 0

    # labels and predicts
    y_label = []
    y_predict = []
    p_label = []
    p_predict = []

    id2class = dataloader.dataset.idx_to_class
    class2id = dataloader.dataset.class_to_idx
    num_classes = dataloader.dataset.num_classes

    # switch to train mode
    model.eval()

    # remap train points to continuous
    import pandas as pd

    metadata = pd.read_csv("data/iwildcam/train.csv")
    remap = {
        location_remapped: y_point
        for y_point, location_remapped in enumerate(
            sorted((metadata["location_remapped"].unique()))
        )
    }

    for idx, (img, target, side_info) in (
        tenumerate(dataloader, ncols=60)
        if utils.is_main_process()
        else enumerate(dataloader)
    ):
        img, c_target = img.to(device), target.to(device)
        p_target = side_info[0]
        remap_p = [remap[i] for i in p_target.tolist()]
        p_target = torch.LongTensor(remap_p).to(device)

        c_output, p_output = model.forward_all(img)

        # measure accuracy and record loss

        # for class loss
        c_loss = criterion(c_output, c_target)
        output = torch.nn.functional.softmax(c_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == c_target)

        y_correct_cnt += correct.item()
        y_label.extend(c_target.int().tolist())
        y_predict.extend(predict.int().tolist())

        y_losses.append(c_loss.item())

        # for point loss
        p_loss = criterion(p_output, p_target)
        output = torch.nn.functional.softmax(p_output, dim=1)
        predict = torch.argmax(output, dim=-1)
        correct = torch.sum(predict == p_target)

        p_correct_cnt += correct.item()
        p_label.extend(p_target.int().tolist())
        p_predict.extend(predict.int().tolist())

        p_losses.append(p_loss.item())

        loss = c_loss + p_loss
        losses.append(loss.item())

        total_cnt += c_target.size(0)

    str_report = classification_report(y_label, y_predict)
    target_names = [id2class[i] for i in range(num_classes)]
    json_report = classification_report(y_label, y_predict, output_dict=True)
    json_report = {
        k + "/" + (target_names[int(k)] if "0" <= k[0] <= "9" else ""): v
        for k, v in json_report.items()
    }

    p_str_report = classification_report(p_label, p_predict)
    p_json_report = classification_report(p_label, p_predict, output_dict=True)

    y_macro_f1 = f1_score(y_label, y_predict, average="macro")
    p_macro_f1 = f1_score(p_label, p_predict, average="macro")

    return {
        "macro_f1": y_macro_f1,
        "loss": np.mean(y_losses),
        "acc": y_correct_cnt * 1.0 / total_cnt,
        "p_macro_f1": p_macro_f1,
        "p_loss": np.mean(p_losses),
        "p_acc": p_correct_cnt * 1.0 / total_cnt,
        "total_loss": np.mean(losses),
        "str_report": str_report,
        "json_report": json_report,
        "p_str_report": p_str_report,
        "p_json_report": p_json_report,
        "y_label": y_label,
        "y_preidct": y_predict,
        "p_label": p_label,
        "p_preidct": p_predict,
    }
