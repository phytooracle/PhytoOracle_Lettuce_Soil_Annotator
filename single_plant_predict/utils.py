from dgcnn import DGCNN
import numpy as np
import torch


def normalize_points(points):
    mod1 = points.mean(axis=0)

    points = points - mod1
    mod2 = np.linalg.norm(points, axis=1).max()
    points /= mod2

    return points, mod1 , mod2

def reverse_normalization(points, mod1, mod2):
    points = points*mod2
    points = points + mod1
    return points

def knn(point, sampled_points, sampled_labels, K, scored_knn=True):
    distances = np.sqrt(np.sum(np.power(sampled_points - point, 2), axis=1))

    if not scored_knn:
        return np.argmax(np.bincount(sampled_labels[np.argsort(distances)[:K]]))

    scores = np.zeros(2)
    for idx in np.argsort(distances)[:K]:
        scores[sampled_labels[idx]] += 1/(distances[idx]+1e-10)

    return scores.argmax()