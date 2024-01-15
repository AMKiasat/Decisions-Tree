from sklearn import datasets
import numpy as np
import math


def entropy_calculator(x):
    entropy = 0
    for i in x:
        if i > 0:
            entropy += -1 * (i * math.log2(i))
    return entropy


def spliter(feature, threshold):
    left_indices = np.where(feature <= threshold)[0]
    right_indices = np.where(feature > threshold)[0]
    return left_indices, right_indices


def label_counter(indices, y):
    if len(indices) > 0:
        y0 = np.where(y == 0)
        y1 = np.where(y == 1)
        y2 = np.where(y == 2)
        count0 = 0
        count1 = 0
        count2 = 0
        for i in indices:
            if i in y0[0]:
                count0 += 1
            elif i in y1[0]:
                count1 += 1
            elif i in y2[0]:
                count2 += 1
        return [count0 / len(indices), count1 / len(indices), count1 / len(indices)]
    else:
        return [0]


def find_best_split(x, y, y_entropy):
    best_gain_ratio = [0, 0, 0]  # [gain_ratio, feature, value]
    for feature in range(len(x.T)):
        selected_spliter = []
        for value in x.T[feature]:
            # print(value)
            if value not in selected_spliter:
                selected_spliter.append(value)
                left_indices, right_indices = spliter(x.T[feature], value)
                entropy = len(left_indices) * entropy_calculator(label_counter(left_indices, y))
                entropy += len(right_indices) * entropy_calculator(label_counter(right_indices, y))
                entropy /= len(y)
                gain = y_entropy - entropy
                split = entropy_calculator([len(left_indices) / len(y), len(right_indices) / len(y)])
                if split > 0:
                    gain_ratio = gain / split
                else:
                    gain_ratio = gain
                if gain_ratio > best_gain_ratio[0]:
                    best_gain_ratio = [gain_ratio, feature, value]
                    # print(best_gain_ratio)
    return best_gain_ratio


def grow_tree(x, y):
    size = len(y)
    label_count = [np.count_nonzero(y == 0) / size,
                   np.count_nonzero(y == 1) / size,
                   np.count_nonzero(y == 2) / size]
    print(find_best_split(x, y, entropy_calculator(label_count)))


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    label = iris.target
    for i in range(len(data)):
        print(i, data[i])
    grow_tree(data, label)
