import os
import torch
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

from correlation import calculate_loss_weights


def count_for_terrainc(metadatapath="data/terrainc/metadata.csv"):
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
    # point2data, label2data = [], []

    metadata: pd.DataFrame = pd.read_csv(metadatapath, index_col=0)
    labels = sorted(metadata["y"].unique())
    points = sorted(metadata["location"].unique())
    point2label_dist = [[0] * len(labels) for _ in range(len(points))]
    label2point_dist = [[0] * len(points) for _ in range(len(labels))]

    point2labels, label2points = [set() for _ in range(len(points))], [
        set() for _ in range(len(labels))
    ]
    for item in metadata.itertuples():
        p, y = item[2], item[3]
        point2label_dist[p][y] += 1
        label2point_dist[y][p] += 1
        point2labels[p].add(y)
        label2points[y].add(p)

    point2labelcnt = [
        sum(cnt > 0 for cnt in point2label_dist[i]) for i in range(len(points))
    ]
    label2pointcnt = [
        sum(cnt > 0 for cnt in label2point_dist[i]) for i in range(len(labels))
    ]

    point2label_dist = np.asarray(point2label_dist)
    label2point_dist = np.asarray(label2point_dist)

    return (
        points,
        labels,
        point2label_dist,
        label2point_dist,
        point2labelcnt,
        label2pointcnt,
        metadata,
    )


def quantitative_correlation_analysis_terrainc(metadata):
    y_label = metadata["y"].tolist()
    y_point = metadata["location"].tolist()
    mi = mutual_info_score(y_label, y_point)
    ami = adjusted_mutual_info_score(y_label, y_point)
    nmi = normalized_mutual_info_score(y_label, y_point)

    print(f"mi: {mi}\nami: {ami}\nnmi: {nmi}")


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
    # plt.hist(point_entropy_of_labels, len(point_entropy_of_labels),  histtype='step', cumulative=True)  # , density=True
    # plt.savefig('pic/point_entropy_of_species_distribution.png')
    # plt.clf()

    # plt.hist(labels_entropy_of_points, len(labels_entropy_of_points),  histtype='step', cumulative=True)  # , density=True
    # plt.savefig('pic/species_entropy_of_points_distribution.png')
    # plt.clf()

    return point_entropy_of_labels, labels_entropy_of_points


if __name__ == "__main__":
    (
        points,
        labels,
        point2label_dist,
        label2point_dist,
        point2labelcnt,
        label2pointcnt,
        metadata,
    ) = count_for_terrainc()

    quantitative_correlation_analysis_terrainc(metadata)

    point_entropy_of_labels, labels_entropy_of_points = calculate_entropy(
        label2point_dist, point2label_dist
    )

    def shifted_softmax_loss_weight(wc, wp):
        return torch.softmax(torch.Tensor([wc + 0.4 * wp, wp]), dim=-1)

    def softmax_loss_weight(wc, wp):
        return torch.softmax(torch.Tensor([wc, wp]), dim=-1)

    def original_loss_weight(wc, wp):
        return wc, wp

    loss_weight_method = original_loss_weight

    # calculate_loss_weights(point2label_dist, labels_entropy_of_points, point_entropy_of_labels, labels, points, threshold_ratio, loss_weight_method)
    for shift in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        sslw = lambda wc, wp: torch.softmax(torch.Tensor([wc + shift * wp, wp]), dim=-1)
        calculate_loss_weights(
            metadata,
            point2label_dist,
            labels_entropy_of_points,
            point_entropy_of_labels,
            labels,
            points,
            0,
            sslw,
            f"shifted_softmax_weight_{shift}p",
        )
