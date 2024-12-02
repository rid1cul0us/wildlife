import os
import json
import shutil
import numpy as np
import pandas as pd

# needs iwildcam metadata file 'metadata.csv' and 'categories.csv' and refined labels csvfile 'labels_refine.csv'

metadata_filepath = "/data/wilds/iwildcam_v2.0/metadata.csv"
categories_filepath = "/data/wilds/iwildcam_v2.0/categories.csv"
label_filter_filepath = "/root/wildlife/dataset/rebuild_iwildcam/labels_refine.csv"

out_put_dir = "/root/wildlife/dataset/rebuild_iwildcam"
image_folder_basepath = "/data/realiwildcam"

# make category_id2y and category_id2name
category_id2y = {}
category_id2name = {}
fd = open(categories_filepath)
fd.readline()
for line in fd.readlines():
    y, id, name = line.split(",")
    category_id2y[id] = y
    category_id2name[id] = name.strip()


def dist(class_filter=None, splits=["id_test", "id_val", "train", "test", "val"]):
    if not os.path.exists(out_put_dir):
        os.mkdir(out_put_dir)

    # count then concat 'y', 'name' and output
    def count(data: pd.DataFrame, output_path: str, class_filter=None):
        data = data["category_id"].value_counts().reset_index()
        data.columns = ["category_id", "count"]

        data["y"] = -1
        data["name"] = "unkonwn"
        for i, r in data.iterrows():
            data.loc[i, "y"] = category_id2y[str(r["category_id"])]
            data.loc[i, "name"] = category_id2name[str(r["category_id"])]

        data = data[data["count"] > 0]

        if class_filter:
            data["y"] = data["y"].astype("int")
            data = data[data["y"].isin(class_filter)]

            # remap 'y' to continuous numbers
            remap = {y: y_remapped for y_remapped, y in enumerate(sorted(class_filter))}
            data["y"] = data["y"].map(remap)

        data.to_csv(os.path.join(out_put_dir, output_path), index=False)

    meta = pd.read_csv(metadata_filepath)
    meta.drop(
        meta.columns[0], axis=1, inplace=True
    )  # remove unnamed index in metadata.csv
    meta["category_id"] = meta["category_id"].astype("category")

    # count all
    # count(meta, "dist.csv", class_filter)

    if class_filter:
        meta = meta[meta["y"].isin(class_filter)]

        # remap 'y' to continuous numbers
        remap = {y: y_remapped for y_remapped, y in enumerate(sorted(class_filter))}
        meta["y"] = meta["y"].map(remap)

    # count each split
    for split in splits:
        part = meta[meta.where(meta["split"] == split)["category_id"].notnull()].copy()

        count(part, f"dist_{split}.csv", class_filter)

        part.to_csv(os.path.join(out_put_dir, f"{split}.csv"), index=False)


def build_image_folder(csvfile, split, y2name):
    global image_folder_basepath
    data = pd.read_csv(csvfile)
    for name in y2name.values():
        os.makedirs(f"{image_folder_basepath}/{split}/{name}", exist_ok=True)
    for i in range(len(data)):
        name = y2name[int(data.loc[i, "y"])]
        from_path = f"/data/wilds/iwildcam_v2.0/train/{data.loc[i, 'filename']}"
        dest_path = f"{image_folder_basepath}/{split}/{name}/{data.loc[i, 'filename']}"
        # print(f"cp {from_path} {dest_path}")
        shutil.copy(from_path, dest_path)


if __name__ == "__main__":
    splits = ["train", "test"]

    dist(splits=splits)
    train = pd.read_csv(os.path.join(out_put_dir, "dist_train.csv"))
    test = pd.read_csv(os.path.join(out_put_dir, "dist_test.csv"))

    train = train[train["count"] >= 50]
    test = test[test["count"] > 0]
    label_filter = pd.read_csv(label_filter_filepath)
    name2id = {name: id for id, name in category_id2name.items()}
    label_filter = [
        int(category_id2y[name2id[label]]) for label in label_filter["raw_lable"]
    ]
    class_filter = (
        set(train["y"].tolist())
        .intersection(test["y"].tolist())
        .intersection(label_filter)
    )
    print(len(class_filter))
    dist(class_filter, splits=splits)

    train = pd.read_csv(os.path.join(out_put_dir, "dist_train.csv"))
    test = pd.read_csv(os.path.join(out_put_dir, "dist_test.csv"))

    y2name = {}
    for i in range(len(train)):
        y2name[int(train.loc[i, "y"])] = train.loc[i, "name"]
        # print(int(train.loc[i, "y"]))
    print(y2name)

    # for i in range(len(test)):
    #     assert (
    #         int(test.loc[i, "y"]) in y2name.keys()
    #     ), f'label {test.loc[i, "y"]} is not in train'
    #     assert y2name[int(test.loc[i, "y"])] == test.loc[i, "name"]

    build_image_folder(os.path.join(out_put_dir, "train.csv"), "train", y2name)
    build_image_folder(os.path.join(out_put_dir, "test.csv"), "test", y2name)

    # ["id_test", "id_val", "train", "test", "val"]
