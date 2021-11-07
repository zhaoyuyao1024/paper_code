from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# 导入数据
data = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
label = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(8))

# 提取胰岛素和皮脂厚度中非零数据
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

# 用非零数据的中值替换缺失数据
for i in range(768):
    if (data[:, 3][i] ==0):
        data[:, 3][i] = np.mean(skin)
    if (data[:, 4][i]==0):
        data[:, 4][i] = np.mean(insulin)
data[:, 3][data[:, 3] == 0] = np.median(skin)+np.random.randint(-10, 10, 1)
data[:, 4][data[:, 4] == 0] = np.median(insulin)+np.random.randint(-30, 30, 1)


# 划分训练集与测试集，训练集为前60%，测试集为后40%
x_train = data[:461]
x_test = data[461:]
y_train = label[:461]
y_test = label[461:]

x_scaled_train = StandardScaler().fit_transform(x_train)
x_scaled_test = StandardScaler().fit_transform(x_test)
# grid = GridSearchCV(LogisticRegression(), param_grid={"C": [0.01, 0.1, 1, 10, 100]},
                   # cv=3, scoring= 'accuracy')
# grid.fit(x_scaled_train, y_train)
# print("最优参数为 %s " % grid.best_params_)

model = LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=500).fit(x_scaled_train, y_train)

y_predict = model.predict(x_scaled_test)
y_true = y_test

classes = list(set(y_true))
classes.sort()
confusion = confusion_matrix(y_predict, y_true)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.title('confusion matrix')
plt.xlabel('prediction')
plt.ylabel('label')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()
print("训练集精度:{:.4f}".format(model.score(x_scaled_train, y_train)))
print("测试集精度:{:.4f}".format(model.score(x_scaled_test, y_test)))