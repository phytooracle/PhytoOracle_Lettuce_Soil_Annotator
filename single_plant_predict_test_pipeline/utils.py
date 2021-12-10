from dgcnn import DGCNN
import numpy as np
import torch


def normalize_points(points):
    points = points - points.mean(axis=0)
    points /= np.linalg.norm(points, axis=1).max()

    return points

def knn(point, sampled_points, sampled_labels, K, scored_knn=True):
    distances = np.sqrt(np.sum(np.power(sampled_points - point, 2), axis=1))

    if not scored_knn:
        return np.argmax(np.bincount(sampled_labels[np.argsort(distances)[:K]]))

    scores = np.zeros(2)
    for idx in np.argsort(distances)[:K]:
        scores[sampled_labels[idx]] += 1/(distances[idx]+1e-10)

    return scores.argmax()