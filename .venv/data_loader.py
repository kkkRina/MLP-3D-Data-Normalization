import math
import random
import numpy as np

class loader:
    def __init__(self,
                 dim=2,
                 trainPercent=85.0,
                 a=1.0, b=2.0, c=0.0):
        self.__tp = trainPercent
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c
        self.__tr, self.__ts = self.__loadData()

    def __loadData(self):
        data = self.__get2DData() if self.dim == 2 else self.__get3DData()
        ln = len(data)
        lnts = int(ln * (1 - self.__tp / 100))
        lntr = ln - lnts

        random.shuffle(data)
        return sorted(data[:lntr]), sorted(data[lntr:])

    def __get2DData(self):
        # Генерация данных для y = cos(x)
        return [
            [
                [i / 10],
                [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    def __get3DData(self):

        # Генерация данных для z = ax**2 + by + c
        data = [
            [
                [x / 10 , y / 10],
                #[self.a * x**2 + self.b * y + self.c + random.uniform(-0.05, 0.05)]
                [x * math.sin(self.a * x / 10 + self.b * y / 10) + self.c + random.uniform(-0.05, 0.05)]
            ]
            for x in range(-15, 16)
            for y in range(-15, 16)
        ]
        return data
        #pass


    def getTrainInp(self):
        return np.array([i[0] for i in self.__tr])

    def getTrainOut(self):
        return np.array([i[1] for i in self.__tr])

    def getTestInp(self):
        return np.array([i[0] for i in self.__ts])

    def getTestOut(self):
        return np.array([i[1] for i in self.__ts])
