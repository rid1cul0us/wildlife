import os
import torch
import matplotlib
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

from dataset.IWildCam import IWildCam

sns.set(
    context="paper",
    palette="deep",
    style="white",
    font="sans-serif",
    font_scale=1,
    color_codes=True,
    rc=None,
)


def load_metadata(categories_filepath, metadata_filepath):
    # make name2y and y2name
    name2y = {}
    y2name = {}
    fd = open(categories_filepath)
    fd.readline()
    for line in fd.readlines():
        y, _, name = line.strip().split(",")
        y = int(y)
        if y > 181:
            break
        name2y[name] = y
        y2name[y] = name

    metadata = pd.read_csv(metadata_filepath)
    metadata.drop(
        metadata.columns[0], axis=1, inplace=True
    )  # remove unnamed index in metadata.csv
    metadata["y"] = metadata["y"].astype("category")
    metadata["location_remapped"] = metadata["location_remapped"].astype("category")

    # draw sample class distribution

    # sample amount analysis
    groupby_y = metadata.groupby("y")
    samples_count = {name: len(groupby_y.get_group(y)) for y, name in y2name.items()}

    plt.figure(figsize=(6,) * 2)
    sns.histplot(
        x=samples_count.keys(),
        weights=samples_count.values(),
        bins=len(samples_count.keys()),
        kde=False,
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Category")
    plt.ylabel("Count")
    # plt.show()

    return metadata


def count(metadata):
    # point2labels[i]:List  species observed at the point
    # label2points[i]:List  points the species presented
    point2labels, label2points = [], []

    # point2label_dist[i]:List  species distribution observed at the point
    # label2point_dist[i]:List  points distribution the species presented
    point2label_dist, label2point_dist = [], []

    # point2labelcnt[i]:int     species count observed at the point
    # label2pointcnt[i]:int     points count the species presented
    point2labelcnt, label2pointcnt = [], []

    # point2data[i]:DataFrame samples index by points
    # label2data[i]:DataFrame samples index by species
    point2data, label2data = [], []

    # location remapping for discrete point number in train split if needed
    remap = {
        location_remapped: y_point
        for y_point, location_remapped in enumerate(
            sorted((metadata["location_remapped"].unique()))
        )
    }
    metadata = metadata.copy()
    metadata["location_remapped"] = (
        metadata["location_remapped"].map(remap).astype(np.int32)
    )

    points = sorted(metadata["location_remapped"].unique())
    labels = sorted(metadata["y"].unique())

    for point in points:
        animal = metadata[metadata["location_remapped"] == point]
        point2labels.append(sorted(animal["y"].unique()))
        point2labelcnt.append(len(point2labels[-1]))
        point2data.append(animal)
        cnt = [0] * len(labels)
        for y in animal["y"]:
            cnt[y] += 1
        point2label_dist.append(cnt)

    for label in labels:
        animal = metadata[metadata["y"] == label]
        label2points.append(sorted(animal["location_remapped"].unique()))
        label2pointcnt.append(len(label2points[-1]))
        label2data.append(animal)
        cnt = [0] * len(points)
        for point in animal["location_remapped"]:
            cnt[point] += 1
        label2point_dist.append(cnt)

    point2label_dist = np.asarray(point2label_dist)
    label2point_dist = np.asarray(label2point_dist)

    return (
        points,
        labels,
        point2label_dist,
        label2point_dist,
        point2labelcnt,
        label2pointcnt,
    )


## cluster
def point_cluster(point2label_dist):
    tsne = TSNE(
        n_components=2, init="pca", random_state=0, perplexity=5
    )  # n_components 为什么选10， 因为75%的点位类别数量不超过10
    result = tsne.fit_transform(point2label_dist)
    plt.scatter(result[:, 0], result[:, 1])
    # plt.show()


## draw point2labelcnt
def draw_point2labelcnt(point2labelcnt):
    plt.hist(
        point2labelcnt, max(point2labelcnt), histtype="step", cumulative=True
    )  # density=True,
    plt.xlabel("species count of each point")
    plt.ylabel("point count")
    plt.title("number of points with the label count of each point")
    plt.xlim(0, max(point2labelcnt))
    plt.savefig("pic/number_of_points_with_the_label_count_of_each_point.pdf", dpi=300)
    plt.clf()

    # sns.histplot(data=point2labels, cumulative=True, stat='density', element='step')
    # fig = sns.displot(data=point2labels, cumulative=True, kind='kde', rug=True)
    # fig.set_ylim(0, 1)


## draw label2ponitcnt
def draw_label2ponitcnt(label2pointcnt):
    plt.hist(
        label2pointcnt, max(label2pointcnt), histtype="step", cumulative=True
    )  # density=True, cumulative=True, label='Empirical'
    plt.xlabel("point count of each species")
    plt.ylabel("species count")
    plt.title("number of labels with the number of presented points of each species")
    plt.xlim(0, max(label2pointcnt))
    plt.savefig(
        "pic/number_of_labels_with_the_number_of_presented_points_of_each_species.pdf",
        dpi=300,
    )
    plt.clf()


## quantitative correlation analysis
def quantitative_correlation_analysis(metadata):
    y_label = metadata["y"].tolist()
    y_point = metadata["location_remapped"].tolist()
    mutual_info_score(y_label, y_point)
    ami = adjusted_mutual_info_score(y_label, y_point)
    nmi = normalized_mutual_info_score(y_label, y_point)

    print(f"ami: {ami}\nnmi: {nmi}")

    ## test set
    test_y_label = metadata[metadata["split"] == "test"]["y"].tolist()
    test_y_point = metadata[metadata["split"] == "test"]["location_remapped"].tolist()
    mi = mutual_info_score(test_y_label, test_y_point)

    ## val set
    # split = 'val'
    # adjusted_mutual_info_score(metadata[metadata['split'] == split]['y'].tolist(), metadata[metadata['split'] == split]['location_remapped'].tolist())

    # y = '24' # 48 49
    # adjusted_mutual_info_score(metadata[metadata['y'] == y]['y'].tolist(), metadata[metadata['y'] == y]['location_remapped'].tolist())


## qualitative correlation analysis
def qualitative_correlation_analysis(
    species2pointcnt, point2speciescnt, labels, points
):
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]

    fig, ax = plt.subplots()
    ax1: matplotlib.axes.Axes = ax
    length = 9
    label_point_count_pos = 1
    point_label_count_pos = 8

    point_bins = [0, 20, 40, 60, 80, len(points)]
    hist, bin_edges = np.histogram(species2pointcnt, bins=point_bins)
    for i in range(len(hist)):
        print(f"{point_bins[i]:3d},{point_bins[i+1]-1:3d}: {hist[i]:2d}")
        ax1.bar(
            label_point_count_pos,
            hist[i],
            bottom=np.sum(hist[:i]),
            width=0.8,
            align="center",
            # label=f'{str(point_bins[i]):>4s}-{str(point_bins[i+1]):>4s}: {str(hist[i]):>4s}')
            label=f"{point_bins[i]}-{point_bins[i+1]-1}: {hist[i]}",
        )

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        title="unique location of each species",
        mode="center",
        fontsize=10,
    )
    ax1.set_ylabel("cumulative number of species", fontsize=12)
    ax1.set_ylim(20, 200)

    ax2: matplotlib.axes.Axes = ax1.twinx()
    # ax2.set_prop_cycle(plt.rcParams['axes.prop_cycle'][len(hist):]) # change color
    # color_list = ['#9AC9DB', '#FFBE7A', '#FA7F6F', '#82B0D2', '#E7DAD2', '#8ECFC9',] # too light
    color_list = [
        "#934B43",
        "#D76364",
        "#EF7A6D",
        "#F1D77E",
        "#B1CE46",
        "#63E398",
    ]
    ax2.set_prop_cycle(plt.cycler(color=color_list))
    species_bins = [0, 10, 20, 30, len(labels)]
    hist, bin_edges = np.histogram(point2speciescnt, bins=species_bins)
    for i in range(len(hist)):
        print(f"{species_bins[i]:3d},{species_bins[i+1]-1:3d}: {hist[i]:2d}")
        ax2.bar(
            point_label_count_pos,
            hist[i],
            bottom=np.sum(hist[:i]),
            width=0.8,
            align="center",
            # label=f'{str(species_bins[i]):>2s}-{str(species_bins[i+1]):>3s}: {str(hist[i]):>3s}'
            # TO DO: align str length
            label=f"{species_bins[i]}-{species_bins[i+1]-1}: {hist[i]}",
        )

    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        title="unique species of each location",
        mode="center",
        fontsize=10,
    )
    ax2.set_ylabel("cumulative number of locations", fontsize=12)
    ax2.set_ylim(50, 350)

    xtick_labels = [
        "",
    ] * length
    xtick_labels[label_point_count_pos] = "unique location count of each species"
    xtick_labels[point_label_count_pos] = "unique species count of each location"
    plt.xticks(range(length), xtick_labels)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    # ax2.set_xticklabels(xtick_labels, fontsize=14)
    # plt.xlim(0, 4)
    plt.title("Analysis of Correlation between Species and Locations", fontsize=10)
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.5)

    plt.savefig("pic/Correlation_Analysis.pdf", dpi=300)
    plt.show()


def calculate_entropy(label2point_dist, point2label_dist):
    point_entropy_of_labels = []
    os.makedirs("pic/point_distribution/", exist_ok=True)
    for y, point_distribution in enumerate(label2point_dist):
        point_entropy = scipy.stats.entropy(point_distribution)
        point_entropy_of_labels.append(point_entropy)
        # plt.plot(range(len(points)), prob)
        # plt.savefig(f'./pic/point_distribution/{y}_{category_id2name[y2category_id[y]]}.png')
        # plt.clf()

    # point_entropy_of_labels
    labels_entropy_of_points = []
    os.makedirs("pic/label_distribution/", exist_ok=True)
    for point, label_distribution in enumerate(point2label_dist):
        label_entropy = scipy.stats.entropy(label_distribution)
        labels_entropy_of_points.append(label_entropy)
        # plt.plot(range(len(labels)), prob)
        # plt.savefig(f'./pic/label_distribution/{point}.png')
        # plt.clf()

    # sns.barplot(entropy_list)
    plt.hist(
        point_entropy_of_labels,
        len(point_entropy_of_labels),
        histtype="step",
        cumulative=True,
    )  # , density=True
    plt.savefig("pic/point_entropy_of_species_distribution.pdf", dpi=300)
    plt.clf()

    plt.hist(
        labels_entropy_of_points,
        len(labels_entropy_of_points),
        histtype="step",
        cumulative=True,
    )  # , density=True
    plt.savefig("pic/species_entropy_of_points_distribution.pdf", dpi=300)
    plt.clf()

    return point_entropy_of_labels, labels_entropy_of_points


def check_threshold(
    metadata,
    labels,
    points,
    point2label_dist,
    labels_entropy_of_points,
    threshold_ratio,
):
    threshold = len(metadata) / len(labels) * threshold_ratio
    after_threshold = []
    for point in points:
        if np.sum(point2label_dist[point]) > threshold:
            if labels_entropy_of_points[point] == 0:
                print(point)
                print(point2label_dist[point])
            after_threshold.append(labels_entropy_of_points[point])

    # print(after_threshold)
    return after_threshold


def calculate_loss_weights(
    metadata,
    point2label_dist,
    labels_entropy_of_points,
    point_entropy_of_labels,
    labels,
    points,
    threshold_ratio,
    loss_weight_method,
    save_name,
):
    # normlization
    # print(point_entropy_of_labels)
    # min_max_normlization = MinMaxScaler(feature_range=(0.1, 1.1)) # MinMaxScaler()
    min_max_normlization = MinMaxScaler(feature_range=(0.1, 1))  # MinMaxScaler()
    point_entropy = min_max_normlization.fit_transform(
        np.asarray(point_entropy_of_labels).reshape((-1, 1))
    )
    label_entropy = min_max_normlization.fit_transform(
        np.asarray(labels_entropy_of_points).reshape((-1, 1))
    )
    point_entropy = np.squeeze(point_entropy)
    label_entropy = np.squeeze(label_entropy)
    # print(point_entropy)

    threshold = len(metadata) / len(labels) * threshold_ratio
    loss_weights = np.zeros((len(labels), len(points), 2))  # (wc, wp)
    for y in labels:
        for point in points:
            if np.sum(point2label_dist[point]) <= threshold:
                loss_weights[y][point] = (1, 0)
            wc = 1 - label_entropy[point]
            wp = 1 - point_entropy[y]
            wc, wp = loss_weight_method(wc, wp)
            loss_weights[y][point] = (wc, wp)

    np.save(save_name, loss_weights)
    # loss_weights = np.load('loss_weights.npy')


def normlize(dist):
    mean, std = np.mean(dist), np.std(dist)
    dist = (dist - mean) / std
    return dist


def normlize_shift(dist):
    mean, std = np.mean(dist), np.std(dist)
    dist = (dist - mean) / std
    dist -= np.min(dist)
    return dist


def temperature_softmax(x, temperature=1.0):
    x = x / temperature
    return torch.softmax(x, dim=-1)


def orignal_weights_visualization(weights_path, label2point_dist, points, labels):
    pass


def softmax_weights_visualization(weights_path, label2point_dist, points, labels):
    weights = np.load(weights_path)

    species, count = (
        [],
        IWildCam.dataset_split("data/iwildcam", "train", None, None).class_cnt,
    )
    means, stds = [], []

    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # species only
    for i in labels[1:]:
        size = [weights[i][j][0] for j in points]
        mean, std = np.mean(size), np.std(size)
        means.append(mean)
        species.append(i)
        stds.append(std)
    sns.scatterplot(
        data={
            "species": species,
            "weight": means,
            "std": stds,
            "count": count[1:],
        },
        x="species",
        y="weight",
        size="std",
        hue="count",
        hue_norm=(1, 5000),
        size_norm=(0.05, 0.2),
    )
    # sns.scatterplot(x=list(range(len(weights))), y=means, sizes=stds, legend=False)
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.05, 0.875),
    )
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(f"pic/weights_visualization_{weights_path}.png", dpi=300)
    print(max(count), min(count))
    print(max(means), min(means))
    print(max(stds), min(stds))


# def model_selection_cmp(path='results/model_selection_cmp/resnet50_all_val_07-03-01:35:48_lr_5e-05_batch_size_48_3trials_iwildcam_ERM_78.4_78.9_45.6_33.4'):
# csvfiles = []
# selected = []

# for root, dirs, files in os.walk(path):
#     for filename in files:
#         if filename.endswith('.csv'):
#             csv = pd.read_csv(os.path.join(root, filename))
#             csvfiles.append(csv)
#             csv = csv.copy()

# for it in csv
# for col in csv.columns:
#
#
#
# results = pd.concat(csvfiles, axis=0, keys=[0, 1, 2])
# mean = results.mean(level=1)
# std = results.std(level=1)


# 广泛分布的代表24 48 49

# entropy
# prob = label2point_dist[1] / np.sum(label2point_dist[1])
# entropy = np.sum(np.log2(prob + 1e-9) * prob * (-1))


if __name__ == "__main__":
    # model_selection_cmp(path='results/model_selection_cmp')
    root_dir = "data/iwildcam"
    metadata_filepath = os.path.join(root_dir, "metadata.csv")
    categories_filepath = os.path.join(root_dir, "categories.csv")

    metadata = load_metadata(categories_filepath, metadata_filepath)
    # train split only
    # metadata = metadata[metadata['split'] == 'train']

    (
        points,
        labels,
        point2label_dist,
        label2point_dist,
        point2labelcnt,
        label2pointcnt,
    ) = count(metadata)

    # point cluster
    # point_cluster(point2label_dist)

    # draw
    os.makedirs("pic", exist_ok=True)
    # draw_label2ponitcnt(label2pointcnt)
    # draw_point2labelcnt(point2labelcnt)

    # # remove 'empty' label
    # species2pointcnt = label2pointcnt[1:]
    # point2speciescnt = [cnt - 1 if point2label_dist[p][0] > 0 else cnt for p, cnt in enumerate(point2labelcnt)]
    # remove 'empty 0' and 'motorbycle 172'
    species2pointcnt = label2pointcnt[1:172] + label2pointcnt[173:]
    point2speciescnt = [
        cnt - (point2label_dist[p][0] > 0) - (point2label_dist[p][172] > 0)
        for p, cnt in enumerate(point2labelcnt)
    ]

    # # correlation analysis
    quantitative_correlation_analysis(metadata)
    qualitative_correlation_analysis(species2pointcnt, point2speciescnt, labels, points)

    point_entropy_of_labels, labels_entropy_of_points = calculate_entropy(
        label2point_dist, point2label_dist
    )
    threshold_ratio = 0.1
    check_threshold(
        metadata,
        labels,
        points,
        point2label_dist,
        labels_entropy_of_points,
        threshold_ratio,
    )
    # calculate_loss_weights

    def shifted_softmax_loss_weight(wc, wp):
        return torch.softmax(torch.Tensor([wc + 0.4 * wp, wp]), dim=-1)

    def softmax_loss_weight(wc, wp):
        return torch.softmax(torch.Tensor([wc, wp]), dim=-1)

    def original_loss_weight(wc, wp):
        return wc, wp

    loss_weight_method = original_loss_weight

    # calculate_loss_weights(point2label_dist, labels_entropy_of_points, point_entropy_of_labels, labels, points, threshold_ratio, loss_weight_method)
    # for shift in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # sslw = lambda wc, wp: torch.softmax(torch.Tensor([wc + shift * wp, wp]), dim=-1)
    # calculate_loss_weights(metadata, point2label_dist, labels_entropy_of_points, point_entropy_of_labels, labels, points, threshold_ratio, sslw, f'shifted_softmax_weight_{shift}p')

    # weights_visualization(weights_path=f'{loss_weight_method.__name__}.npy', label2point_dist=label2point_dist, points=points, labels=labels)
