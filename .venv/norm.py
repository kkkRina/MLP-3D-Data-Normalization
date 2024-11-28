import numpy as np


class norm:
    def __init__(self, data):
        self.__min = np.min(data, axis=0)
        self.__max = np.max(data, axis=0)

    def norm(self, x):
        x = np.array(x)
        return 2 * (x - self.__min) / (self.__max - self.__min) - 1

    def denorm(self, y):
        y = np.array(y)
        return (y + 1) / 2 * (self.__max - self.__min) + self.__min

