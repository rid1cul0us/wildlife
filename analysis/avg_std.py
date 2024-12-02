import os
import numpy as np
import pandas as pd

from typing import DefaultDict, List, Iterable


def selection(
    path: str,
    name_filter: str,
    max_iter=500,
    selectors: Iterable[str] = [
        "all_val acc",
        "test acc",
        "all_val macro_f1",
        "test macro_f1",
    ],
    metrics: Iterable[str] = [
        "id_test acc",
        "test acc",
        "id_test macro_f1",
        "test macro_f1",
    ],
):
    """_summary_

    Args:
        path (str): csv outputs base dir.
        name_filter (str): csv filenames filter.
        selectors (Iterable[str], optional): metrics used to select model. Defaults to ['val acc', 'test acc'].
        metrics (Iterable[str], optional): metrics we concern. Defaults to ['id_test acc', 'test acc'].
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print
    metric_path = f"{name_filter}/metric.txt"
    f = open(metric_path, "w")
    f.close()

    def print(*args, **kwargs):
        if os.path.isdir(name_filter):
            builtin_print(
                *args,
                **kwargs,
                file=open(metric_path, "a+" if os.path.exists(metric_path) else "w"),
            )
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    if not os.path.exists(path):
        return
    print(f"{path}({name_filter}):")

    results = DefaultDict(DefaultDict)
    for selector in selectors:
        results[selector] = DefaultDict(list)

    for root, dirs, files in os.walk(path):
        if name_filter in root:
            for filename in files:
                if filename.endswith(".csv"):
                    data = pd.read_csv(os.path.join(root, filename))
                    best_value = DefaultDict(float)
                    best_row = DefaultDict(object)
                    which_epoch = DefaultDict(object)
                    for index, row in data.iterrows():
                        if index + 1 > max_iter:
                            break
                        for selector in selectors:
                            if hasattr(row, selector):
                                if row[selector] > best_value[selector]:
                                    best_value[selector] = row[selector]
                                    best_row[selector] = row
                                    which_epoch[selector] = index + 1
                    for selector in selectors:
                        if selector in best_row.keys():
                            for metric in metrics:
                                if hasattr(best_row[selector], metric):
                                    results[selector][metric].append(
                                        best_row[selector][metric]
                                    )

                    best_model_selector = "val macro_f1"
                    if best_model_selector in best_row.keys():
                        best_epoch = which_epoch[best_model_selector]
                        models_dir = os.path.join(root, "tmodels")
                        for model in os.listdir(models_dir):
                            if f"_epo{best_epoch}_" in model:
                                if not os.path.exists(f"{models_dir}/best_model.pth"):
                                    os.system(
                                        f"cp {models_dir}/{model} {models_dir}/best_model.pth"
                                    )
                                print(f"best_model: {model}")
                                break

    new_name = None
    for selector, result in results.items():
        print(f"{selector} select:")
        ret, seeds = report(**result)
        if not len(ret) or seeds != 3:
            continue
        if os.path.basename(name_filter).count("_") < 15 and selector == "val macro_f1":
            if len(ret) == 4:
                id_acc, test_acc, id_f1, test_f1 = ret
                new_name = f"{name_filter}_{id_acc[0]}({id_acc[1]})_{test_acc[0]}({test_acc[1]})_{id_f1[0]}({id_f1[1]})_{test_f1[0]}({test_f1[1]})"
            else:
                acc, f1 = ret
                new_name = f"{name_filter}_{acc[0]}({acc[1]})_{f1[0]}({f1[1]})"
            while os.path.exists(new_name):
                new_name + new_name + "_"
    if new_name:
        os.rename(name_filter, new_name)


def report(**results):
    ret = []
    result = None
    for metric, result in results.items():
        mean, std = f"{np.mean(result)*100:.1f}", f"{np.std(result)*100:.1f}"
        print(f"{metric} in {len(result)} trials\t{mean}({std})")
        print(result)
        ret.append([mean, std])
    print()
    return ret, len(result) if result else None


if __name__ == "__main__":
    # selection(path='/home/kafm/program/wildlife/results', name_filter='resnet50_val')
    # selection(path='/root/ssd_data/kafm/wildlife/results', name_filter='convnext_tiny')
    # selection(path='/root/ssd_data/kafm/wildlife/results', name_filter='convnext_tiny_all_val')
    # selection(path='/home/kafm/program/wildlife/results/resnet50_224x224_4trials_all_valacc_selfaugment_seed+rank_10ep', name_filter='resnet50_all_val')
    # selection(path='results', name_filter='results/resnet50_all_val_05-25_lr_5e-05_batch_size_48_3trials_randaugment_ptw_1')
    # selection(path='results', name_filter='convnext_tiny_all_val')
    selection(
        path="results",
        max_iter=500,
        name_filter=f"results/report/bw/PU/vit/swinv2_tiny_randaug_PU_3trials",
        selectors=[
            "id_val acc",
            "val acc",
            "all_val acc",
            "test acc",
            "id_val macro_f1",
            "val macro_f1",
            "all_val macro_f1",
            "test macro_f1",
        ],
    )

    # PU(original) & ResNet50 & 45.4\footnotesize$\pm{1.2}$ & 78.9\footnotesize$\pm{0.5}$ & 33.4\footnotesize$\pm{0.3}$ & 78.4\footnotesize$\pm{1.1}$  \\
    # DU(softmax) & ResNet50 & 45.2\footnotesize$\pm{0.7}$ & 77.1\footnotesize$\pm{0.1}$ & 33.2\footnotesize$\pm{0.6}$ & 77.7\footnotesize$\pm{0.4}$  \\

    # for model selection evaluation
    # selection(path='results', max_iter=500, name_filter='results/model_selection_cmp/resnet50_all_val_07-03-01:35:48_lr_5e-05_batch_size_48_3trials_iwildcam_ERM', selectors=['id_val acc', 'val acc', 'all_val acc', 'test acc', 'id_val macro_f1', 'val macro_f1', 'all_val macro_f1', 'test macro_f1'], metrics=['id_test acc', 'test acc', 'id_test macro_f1', 'test macro_f1'])

    # iid val  ood val   robust val
    #

    # basedir = 'results/report/bw/DU/ptw_ablation'
    # for f in os.listdir(basedir):
    # selection(path='results', max_iter=500, name_filter=f'{basedir}/{f}',\
    # selectors=['id_val acc', 'val acc', 'all_val acc', 'test acc', 'id_val macro_f1', 'val macro_f1', 'all_val macro_f1', 'test macro_f1'])

    pass

"""

resnet50 224x224 3 trials
val acc select:
id_test acc     75.7(0.6)
test acc        71.5(3.4)

test acc select:
id_test acc     76.0(0.4)
test acc        75.1(0.7)


convnext_tiny 224x224 8 trials
val acc select:
id_test acc     78.8(0.7)
test acc        77.9(0.9)

test acc select:
id_test acc     77.7(1.8)
test acc        79.1(0.6)


convnext_tiny 224x224 3 trials
all_val acc select:
id_test acc     78.4(0.4)
test acc        78.8(0.5)

test acc select:
id_test acc     78.0(1.6)
test acc        79.2(0.6)



/home/kafm/program/wildlife/results/resnet50_224x224_4trials_all_valacc_selfaugment_seed+rank_10ep(resnet50_all_val):
all_val acc select:
id_test acc in 4 trials 76.7(0.7)
test acc in 4 trials    70.6(2.1)
id_test macro_f1 in 1 trials    42.9(0.0)
[0.42861]
test macro_f1 in 1 trials       31.3(0.0)
[0.3131]

val acc select:

test acc select:
id_test acc in 4 trials 75.8(1.7)
[0.76576, 0.76613, 0.72946, 0.77103]
test acc in 4 trials    74.3(1.5)
[0.73013, 0.73751, 0.73518, 0.76869]
id_test macro_f1 in 1 trials    38.8(0.0)
[0.38829]
test macro_f1 in 1 trials       28.1(0.0)
[0.28113]

all_val macro_f1 select:
id_test acc in 1 trials 76.7(0.0)
[0.76699]
test acc in 1 trials    75.5(0.0)
[0.75474]
id_test macro_f1 in 1 trials    42.3(0.0)
[0.42276]
test macro_f1 in 1 trials       30.2(0.0)
[0.30171]



448x448
results(resnet50_all_val):
all_val acc select:
id_test acc in 3 trials 78.3(0.4)
[0.78906, 0.77913, 0.78011]
test acc in 3 trials    78.5(0.6)
[0.7775, 0.7925, 0.7839]
id_test macro_f1 in 3 trials    42.4(1.1)
[0.40854, 0.43255, 0.43077]
test macro_f1 in 3 trials       33.2(1.5)
[0.33826, 0.34712, 0.31085]

val acc select:

test acc select:
id_test acc in 3 trials 78.0(0.3)
[0.78354, 0.77716, 0.78011]
test acc in 3 trials    78.9(0.4)
[0.78904, 0.79332, 0.7839]
id_test macro_f1 in 3 trials    42.3(0.7)
[0.41341, 0.42616, 0.43077]
test macro_f1 in 3 trials       32.2(1.0)
[0.31964, 0.33501, 0.31085]

all_val macro_f1 select:
id_test acc in 3 trials 77.8(0.2)
[0.78048, 0.77729, 0.77716]
test acc in 3 trials    78.0(0.4)
[0.77792, 0.78589, 0.77755]
id_test macro_f1 in 3 trials    43.5(1.5)
[0.4151, 0.45039, 0.43979]
test macro_f1 in 3 trials       33.0(0.5)
[0.33743, 0.32577, 0.32596]

test macro_f1 select:
id_test acc in 3 trials 78.2(0.5)
[0.78906, 0.77913, 0.77692]
test acc in 3 trials    78.3(0.7)
[0.7775, 0.7925, 0.77846]
id_test macro_f1 in 3 trials    42.4(1.1)
[0.40854, 0.43255, 0.4304]
test macro_f1 in 3 trials       33.8(0.7)
[0.33826, 0.34712, 0.32925]
"""
