import os
import json
import numpy as np
import pandas as pd


# filepath of metadata.csv and categories.csv in iwildcam2.0 from wilds
metadata_filepath = "data/iwildcam/metadata.csv"
categories_filepath = "data/iwildcam/categories.csv"

# filepath of json result from MegaDetector
# mega_result_file = '/home/kafm/project/wilds/meta/mega_result_1.json'

out_put_dir = "data/iwildcam"


def dist(metadata_filepath, categories_filepath, out_put_dir):
    if not os.path.exists(out_put_dir):
        os.mkdir(out_put_dir)

    # make category_id2y and category_id2name
    category_id2y = {}
    category_id2name = {}
    fd = open(categories_filepath)
    fd.readline()
    for line in fd.readlines():
        y, id, name = line.split(",")
        category_id2y[id] = y
        category_id2name[id] = name.strip()

    # count then concat 'y', 'name' and output
    def count(data: pd.DataFrame, output_path: str):
        data = data["category_id"].value_counts().reset_index()
        data.columns = ["category_id", "count"]

        data["y"] = -1
        data["name"] = "unkonwn"
        for i, r in data.iterrows():
            data.loc[i, "y"] = category_id2y[str(r["category_id"])]
            data.loc[i, "name"] = category_id2name[str(r["category_id"])]

        data.to_csv(os.path.join(out_put_dir, output_path), index=False)

    meta = pd.read_csv(metadata_filepath)
    meta.drop(
        meta.columns[0], axis=1, inplace=True
    )  # remove unnamed index in metadata.csv
    meta["category_id"] = meta["category_id"].astype("category")

    # count all
    count(meta, "dist.csv")

    # count each split
    for split in ["id_test", "id_val", "train", "test", "val"]:
        part = meta[meta.where(meta["split"] == split)["category_id"].notnull()].copy()
        count(part, f"dist_{split}.csv")

        if split == "train":
            # remap 'location_remapped' field to continuous numbers
            remap = {
                location_remapped: y_point
                for y_point, location_remapped in enumerate(
                    sorted((part["location_remapped"].unique()))
                )
            }
            part["y_point"] = part["location_remapped"].map(remap)

        part.to_csv(os.path.join(out_put_dir, f"{split}.csv"), index=False)


def output_check():
    dist, meta = {}, {}
    files = [
        "dist.csv",
        "dist_id_test.csv",
        "dist_id_val.csv",
        "dist_test.csv",
        "dist_train.csv",
        "dist_val.csv",
    ]
    (
        dist["all"],
        dist["id_test"],
        dist["id_val"],
        dist["test"],
        dist["train"],
        dist["val"],
    ) = [pd.read_csv(os.path.join(out_put_dir, f)) for f in files]

    files = ["id_test.csv", "id_val.csv", "test.csv", "train.csv", "val.csv"]
    meta["id_test"], meta["id_val"], meta["test"], meta["train"], meta["val"] = [
        pd.read_csv(os.path.join(out_put_dir, f)) for f in files
    ]

    metadata = pd.read_csv(metadata_filepath)
    metadata.drop(
        metadata.columns[0], axis=1, inplace=True
    )  # remove unnamed index in metadata.csv
    metadata["category_id"] = metadata["category_id"].astype("category")
    meta["all"] = metadata

    # print(dist)

    # for k, v in meta.items() :
    #     print(f'{k}\t:{len(v)}')

    # for k, v in dist.items() :
    #     print(f'{k}\t:{v["count"].sum()}')

    # assert(dist['all']['count'].sum() == len(meta))


# deprecated
def metadata_fix(mega_result_file=""):
    metadata = pd.read_csv(metadata_filepath)
    metadata.drop(
        metadata.columns[0], axis=1, inplace=True
    )  # remove unnamed index in metadata.csv
    metadata["megaType"] = "animal"

    #    all: 217640
    #    mega_result_1.json
    #    threshold      all     metadata
    #       0.20       99862
    #       0.15       96760
    #       0.10       91933
    #       0.05       93422    84839
    #    mega empty but metadata animal:  27624
    empty = set()
    threshold = 0.05
    megaresult = json.load(open(mega_result_file))
    for result in megaresult["images"]:
        isEmpty = True
        for d in result["detections"]:
            if d["conf"] > threshold and d["category"] == "1":
                isEmpty = False
                break
        if isEmpty:
            # empty.add(result['file'].split('/')[-1].split('.')[0])
            empty.add(result["file"].split("/")[-1])
    print(f"threshold: {threshold}  empty: {len(empty)}")

    ## images in meta is part of all images, meta_not_in_mega is 0
    # meta_not_in_mega = 0
    # for fn in metadata.loc[:, 'filename']:
    #     if fn not in all_:
    #         meta_not_in_mega += 1
    # print(f'meta_not_in_mega: {meta_not_in_mega}')

    # check label is 'empty' but mega detects animal
    label_empty_mega_animal = 0
    for i in range(len(metadata)):
        if metadata.loc[i, "image_id"] in empty:
            metadata.loc[i, "megaType"] = "empty"
        elif metadata.loc[i, "y"] == 0:
            label_empty_mega_animal += 1
    print("label_empty_mega_animal: ", label_empty_mega_animal)

    metadata.loc[
        [metadata.loc[i, "filename"] in empty for i in range(len(metadata))], "megaType"
    ] = "empty"

    metadata_animal = metadata.where(metadata["megaType"] == "animal").dropna()
    print(
        "mega empty but metadata animal: ",
        metadata.where((metadata["megaType"] == "empty") & (metadata["y"] != 0))["y"]
        .notnull()
        .sum(),
    )
    fixed = metadata.where(metadata["megaType"] == "empty")["megaType"].notnull()
    metadata.loc[fixed, "y"] = metadata.loc[fixed, "category_id"] = 0

    # TO DO
    # 根据样本数量重新映射连续的 y
    metadata_animal["category_id"] = metadata_animal["category_id"].astype("category")

    metadata_animal.to_csv(
        "/home/kafm/project/wilds/meta/metadata_animals.csv", index=False
    )
    metadata.to_csv("/home/kafm/project/wilds/meta/metadata_fixed.csv", index=False)


if __name__ == "__main__":
    dist(metadata_filepath, categories_filepath, out_put_dir)

    # output_check()

    # metadata_fix()
