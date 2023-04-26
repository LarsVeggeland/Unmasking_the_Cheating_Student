import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix

def get_data(filname : str, column_count : int, max_entries : int=None, seed : int=42) -> tuple:

    df = pd.read_csv(filname)
    labels = df["label"].to_list()
    features = df.iloc[:,1:column_count].to_numpy()

    if max_entries is not None:
        indicies = [i for i in range(len(labels))]
        random.seed(a=seed)
        random.shuffle(indicies)
        if max_entries >= len(labels):
            print(f"The max_entries: {max_entries} is greater than the number of entries in the dataset: {len(labels)}")
            print("Therefore, the full dataset will be returned")
        else:
            indicies = indicies[:max_entries]
            labels = [labels[i] for i in indicies]
            features = [features[i] for i in indicies]

    return (features, labels)

def filter_entries_by_index(features : np.ndarray, labels : np.ndarray, filter : list, positive_match : bool = True) -> tuple:
    filtered_features = []
    filtered_labels = []
    for i, curve in enumerate(features):
        if i in filter and positive_match:
            filtered_features.append(curve)
            filtered_labels.append(labels[i])
        elif i not in filter and not positive_match:
            filtered_features.append(curve)
            filtered_labels.append(labels[i])
    
    return (filtered_features, filtered_labels)


def filter_entries_by_label(features : np.ndarray, labels : np.ndarray, label_val : bool) -> bool:
    filtered_features = []
    filtered_labels = []
    for i, curve in enumerate(features):
        if labels[i] == label_val:
            filtered_features.append(curve)
            filtered_labels.append(labels[i])
    
    return (filtered_features, filtered_labels)


def plot(features, labels, plot_title, lw):
    for i, curve in enumerate(features):
        color = "b" if labels[i] == True else "r"
        author = "same author curve" if labels[i] == True else "different author curve"
        plt.plot(curve, color=color, linewidth = lw)

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.show()


def plot_two_classes(x1, x2, color1, color2, plot_title):
    for curve in x1:
        plt.plot(curve, color=color1)

    for curve in x2:
        plt.plot(curve, color=color2)

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(plot_title)
    plt.show()


if __name__ == '__main__':
    x, y = get_data("data//curves//KoppelOriginal5K.csv", 20)
    #x, y = filter_entries_by_label(x, y, True)
    #x, y = filter_entries_by_index(x, y, [2, 3, 8, 14, 17, 18, 19, 20, 22, 23, 24, 25, 27, 29, 30, 31, 33, 37, 38, 39][19:])
    #x, y = filter_entries_by_index(x, y, [37, 18, 19, 23, 33, 38])
    #x, y = filter_entries_by_index(x, y, [37, 23, 33])
    plot(x, y, "Unmasking Using Words and Sliding Window Sampling" ,0.5)
    #print([i for i, l in enumerate(y) if l])