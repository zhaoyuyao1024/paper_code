import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import *


np.random.seed(0)
def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1/(1 + np.exp(-np.dot(theta.T, x)))

class NNet3:
    # 初始化必要的几个参数
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)
    def _multiplecost(self, X, y):
        # l1是中间层的输出，l2是输出层的结果
        l1, l2 = self._feedforward(X)
        # 计算误差，这里的l2是前面的h
        inner = y * np.log(l2) + (1-y) * np.log(1-l2)
        # 添加符号，将其转换为正值
        return -np.mean(inner)

    # 前向传播函数计算每层的输出结果
    def _feedforward(self, X):
        # l1是中间层的输出
        l1 = sigmoid_activation(X.T, self.theta0).T
        # 为中间层添加一个常数列
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # 中间层的输出作为输出层的输入产生结果l2
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2

    # 传入一个结果未知的样本，返回其属于1的概率
    def predict(self, X):
        _, y = self._feedforward(X)
        return y

    # 学习参数，不断迭代至参数收敛，误差最小化
    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))

        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1
        counter = 0

        for counter in range(self.maxepochs):
            # 计算中间层和输出层的输出
            l1, l2 = self._feedforward(X)

            # 首先计算输出层的梯度，再计算中间层的梯度
            l2_delta = (y-l2) * l2 * (1-l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

            # 更新参数
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate

            counter += 1
            costprev = cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break

if __name__ == '__main__':
    data = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    label = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(8))
    insulin = []
    skin = []
    i = 0
    for i in range(768):
        if (data[:, 4][i] != 0):
            insulin.append(data[:, 4][i])
        if (data[:, 3][i] != 0):
            skin.append(data[:, 3][i])
    insulin = np.sort(insulin)
    skin = np.sort(skin)
    for i in range(768):
        if (data[:, 3][i] == 0):
            data[:, 3][i] = np.median(skin) + np.random.randint(-10, 10, 1)
        if (data[:, 4][i] == 0):
            data[:, 4][i] = np.median(insulin) + np.random.randint(-30, 30, 1)
    x_train = data[:461]
    x_test = data[461:]
    y_train = label[:461]
    y_test = label[461:]
    # 对数据进行归一化处理
    x_scaled_train = StandardScaler().fit_transform(x_train)
    x_scaled_test = StandardScaler().fit_transform(x_test)

    learning_rate = 0.1
    maxepochs = 1000
    convergence_thres = 0.00001
    hidden_units = 10
    model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
                  convergence_thres=convergence_thres, hidden_layer=hidden_units)
    model.learn(x_scaled_train, y_train)
    train_yhat = model.predict(x_scaled_train)[0]
    train_auc = roc_auc_score(y_train,train_yhat)
    print(train_auc)
    yhat = model.predict(x_scaled_test)[0]
    predict = []

    for each in yhat:
        if each > 0.5:
            predict.append(1)
        else:
            predict.append(0)
    auc = roc_auc_score(y_test, yhat)
    print(auc)