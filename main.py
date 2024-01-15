from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import math
import pickle


def entropy_calculator(x):
    entropy = 0
    for i in x:
        if i > 0:
            entropy += -1 * (i * math.log2(i))
    return entropy


# def most_repeated_int(arr):
#     unique_elements, counts = np.unique(arr, return_counts=True)
#     max_count_index = np.argmax(counts)
#     most_repeated = unique_elements[max_count_index]
#
#     return [most_repeated, max_count_index]


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


def grow_tree(x, y, offset_num):
    size = len(y)
    label_count = [np.count_nonzero(y == 0) / size,
                   np.count_nonzero(y == 1) / size,
                   np.count_nonzero(y == 2) / size]
    lc = [np.count_nonzero(y == 0), np.count_nonzero(y == 1), np.count_nonzero(y == 2)]
    # print(y)
    if size - np.max(lc) <= offset_num:
        return np.argmax(lc)
    best_split = find_best_split(x, y, entropy_calculator(label_count))
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for i in range(len(x)):
        if x[i][best_split[1]] <= best_split[2]:
            left_x.append(x[i])
            left_y.append(y[i])
        else:
            right_x.append(x[i])
            right_y.append(y[i])
    # print(best_split[2], len(left_x), len(right_x))
    left = grow_tree(np.array(left_x), np.array(left_y), offset_num)
    right = grow_tree(np.array(right_x), np.array(right_y), offset_num)

    return left, best_split[1], best_split[2], right


def train(data, label, offset_num):
    tree = grow_tree(data, label, offset_num=offset_num)
    print(tree)

    with open('tree.pkl', 'wb') as file:
        pickle.dump(tree, file)


def predict(x, tree):
    if isinstance(tree, np.int64):
        return tree
    elif x[tree[1]] <= tree[2]:
        return predict(x, tree[0])
    else:
        return predict(x, tree[3])


def test(data):
    with open('tree.pkl', 'rb') as file:
        loaded_tree = pickle.load(file)
    prediction = []
    for i in data:
        prediction.append(predict(i, loaded_tree))
    return prediction


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    label = iris.target
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=1)

    train(train_data, train_labels, offset_num=5)

    predictions = test(test_data)
    # print(predictions)
    correct = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            correct += 1
    print(correct, len(test_labels), correct / len(test_labels))
