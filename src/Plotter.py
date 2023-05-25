import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def get_data(filname : str, column_count : int, max_entries : int=None, seed : int=42) -> tuple:

    df = pd.read_csv(filname)
    labels = df["label"].to_numpy()
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
    
    return (np.array(filtered_features), np.array(filtered_labels))


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



def class_avg(labels, features) -> tuple:
    
    avg_same_curves = np.zeros(features.shape[1])
    avg_different_curves = np.zeros(features.shape[1])

    for i, label in enumerate(labels):
        if label:
            avg_same_curves += features[i,:]
        else:
            avg_different_curves += features[i,:]
    
    same_curves_count = np.sum(labels)

    avg_same_curves = avg_same_curves/same_curves_count
    avg_different_curves = avg_different_curves/(labels.shape[0]-same_curves_count)

    return (avg_same_curves, avg_different_curves)


def class_std(labels, features, avg_same, avg_different) -> tuple:

    var_same = np.zeros(features.shape[1])
    var_different = np.zeros(features.shape[1])

    for i, label in enumerate(labels):
        if label:
            var_same += (features[i,:] - avg_same)**2
        else:
            var_different += (features[i,:] - avg_different)**2
    
    same_count = np.sum(labels)
    std_same = np.zeros(var_same.shape[0])
    std_different = np.zeros(var_same.shape[0])

    for i in range(std_same.shape[0]):
        std_same[i] = (var_same[i]/same_count)**0.5
        std_different [i]= (var_different[i])/(labels.shape[0]-same_count)**0.5
    
    return (std_same, std_different)


def plot_avg_and_std(avg, std, color):
    x = range(len(avg))
    y = avg
    plt.plot(x, y, color=color, linewidth=0.5)

    # Makes sure that the plot does not exceed 1, i.e., 100 % 
    cap_at_one = lambda x,y : x+y if x+y < 1 else 1
    plt.fill_between(x, [(y[i]-std[i]) for i in x], [cap_at_one(y[i], std[i]) for i in x], color=color, alpha=0.1)




def stats_by_class(features, labels):
    class_0_indices = np.where(labels == 0)
    class_1_indices = np.where(labels == 1)

    class_0_features = features[class_0_indices]
    class_1_features = features[class_1_indices]

    mean_class_0 = np.mean(class_0_features, axis=0)
    mean_class_1 = np.mean(class_1_features, axis=0)

    std_class_0 = np.std(class_0_features, axis=0)
    std_class_1 = np.std(class_1_features, axis=0)

    return mean_class_0, std_class_0, mean_class_1, std_class_1



def plot_same_and_different_avg_std(features, labels, title):
    #avg_same, avg_different = class_avg(labels, features)
    #std_same, std_different = class_std(labels, features, avg_same, avg_different)
    mean_diff, std_diff, mean_same, std_same = stats_by_class(features, labels)
    print(np.mean(std_same), np.mean(std_diff))
    plot_avg_and_std(mean_same, std_same, "blue")
    plot_avg_and_std(mean_diff, std_diff, "red")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == '__main__':
    #x, y = get_data(r"data/curves/PreliminartTesting/OriginalLexpos.csv", 42)
    x, y = get_data(r"data/curves/PreliminartTesting/OriginalLexpos10KCapBalanced.csv", 42)

    #x, y = filter_entries_by_label(x, y, False)
    #a, b = filter_entries_by_label(a, b, False)

    #y = np.array([True for i in range(y.shape[0])])

    #print(y.shape, b.shape)
    #y = np.concatenate((y, b))
    #x = np.concatenate((x, a), axis=0)

    #print(y.shape, x.shape)

    plot_same_and_different_avg_std(x, y, "Average Same and Different Author Curve (Lexpos)")
    #plot(x, y, "Unmasking Using Lexpos With Default Settings" ,0.5)
    #print([i for i, l in enumerate(y) if l])