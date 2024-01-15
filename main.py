from sklearn import datasets
import numpy as np
import math


def entropy_calculator(x):
    entropy = 0
    for i in x:
        entropy += -1 * (i * math.log2(i))
    return entropy


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    label = iris.target
    size = len(label)
    label_count = [np.count_nonzero(label == 0) / size,
                   np.count_nonzero(label == 1) / size,
                   np.count_nonzero(label == 2) / size]

    print(entropy_calculator(label_count))
