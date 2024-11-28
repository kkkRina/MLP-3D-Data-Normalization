import data_loader as dl
from MLPerceptron import MLP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков
import numpy as np
from norm import norm as n

def plot2D():
    ld = dl.loader(trainPercent=85)
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    tsi = ld.getTestInp()
    tso = ld.getTestOut()

    mlp = MLP(ld, (10,))
    e_tr, e_ts = mlp.learn(epsilon=1e-3, epoches=500)
    e_ts_x = [i for i in range(1, len(e_ts) + 1)]
    f1 = plt.figure(1)
    fa1 = f1.add_subplot(1, 1, 1)
    out = mlp.calc(tri)
    fa1.plot(tri, out, "b-")
    fa1.plot(tri, tro, "r+")

    out = mlp.calc(tsi)
    fa1.plot(tsi, out, "gv")
    fa1.plot(tsi, tso, "y+")

    f2 = plt.figure(2)
    fa2 = f2.add_subplot(1, 1, 1)
    fa2.plot(e_ts_x[300:], e_ts[300:], "b-")
    fa2.plot(e_ts_x[300:], e_tr[300:], "r-")
    plt.show()


def plot3D():
    # Загрузка данных
    ld = dl.loader(dim=3, trainPercent=42)
    tri = np.array(ld.getTrainInp())
    tro = np.array(ld.getTrainOut())
    tsi = np.array(ld.getTestInp())
    tso = np.array(ld.getTestOut())

    # Нормализация данных
    input_norm = n(tri)
    output_norm = n(tro)

    tri_norm = input_norm.norm(tri)
    tro_norm = output_norm.norm(tro)
    tsi_norm = input_norm.norm(tsi)
    tso_norm = output_norm.norm(tso)

    # Создание MLP и обучение
    mlp = MLP(tri_norm, tro_norm, tsi_norm, tso_norm, (25,))
    e_tr, e_ts = mlp.learn(epsilon=1e-3, epoches=1000)

    # Предсказания модели
    pred_train_norm = mlp.predict(tri_norm)
    pred_test_norm = mlp.predict(tsi_norm)

    pred_train = output_norm.denorm(pred_train_norm)
    pred_test = output_norm.denorm(pred_test_norm)

    '''
    # Создание MLP и обучение
    mlp = MLP(ld, (15,))
    e_tr, e_ts = mlp.learn(epsilon=10e-4, epoches=1000)

    # Предсказания модели
    pred_train = mlp.calc(tri)
    pred_test = mlp.calc(tsi)
    '''
    # Разделение координат для графика
    train_x = tri[:, 0]
    train_y = tri[:, 1]
    train_z = tro[:, 0]

    test_x = tsi[:, 0]
    test_y = tsi[:, 1]
    test_z = tso[:, 0]

    pred_train_z = pred_train
    pred_test_z = pred_test

    print("Original Train Output Z range:", tro.min(), tro.max())
    print("Norm Train Output Z range:", tro_norm.min(), tro_norm.max())
    print("Denorm Predictions Z range:", pred_train.min(), pred_train.max())
    print("Predicted Train Output Z range:", pred_train_z.min(), pred_train_z.max())

    # Визуализация 3D-данных
    fig = plt.figure(figsize=(14, 7))

    # График предсказаний и реальных данных
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(train_x, train_y, train_z, color="red", label="Train Data", alpha=0.7)
    ax1.scatter(train_x, train_y, pred_train_z, color="blue", marker="^", label="Train Predictions", alpha=0.7)
    ax1.scatter(test_x, test_y, test_z, color="orange", label="Test Data", alpha=0.7)
    ax1.scatter(test_x, test_y, pred_test_z, color="green", marker="^", label="Test Predictions", alpha=0.7)
    ax1.set_title("3D Data and Predictions")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    ax1.grid()

    # График ошибок
    ax2 = fig.add_subplot(122)
    ax2.plot(e_tr, label="Train Error", color="blue")
    ax2.plot(e_ts, label="Test Error", color="red")
    ax2.set_title("Training and Testing Errors")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()






#plot2D()
plot3D()