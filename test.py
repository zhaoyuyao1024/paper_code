import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import *
from sklearn import metrics
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.random.seed(1)

# 导入数据
data = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
label = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(8))

# 提取出非零数据
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
# print("SkinThickness的均值为%s" % np.mean(skin))
# print("SkinThickness的中值为%s" % np.median(skin))
# print("SkinThickness的众数为%s" % mode(insulin)[0][0])
# print("Insulin的均值为%s" % np.mean(insulin))
# print("Insulin的中值为%s" % np.median(insulin))
# print("Insulin的众数为%s" % mode(insulin)[0][0])
# 用非零数据的中值替换缺失数据
for i in range(768):
    if (data[:, 3][i] ==0):
        data[:, 3][i] = np.median(skin)+np.random.randint(-10, 10, 1)
    if (data[:, 4][i]==0):
        data[:, 4][i] = np.median(insulin)+np.random.randint(-30, 30, 1)

# 划分训练集与测试集，训练集为前60%，测试集为后40%
x_train = data[:461]
x_test = data[461:]
y_train = label[:461]
y_test = label[461:]

# 通过特征选择筛选出k个最佳特征
# print(SelectKBest(chi2, k=6).fit_transform(x_train, y_train)[0])

# 对数据进行归一化处理
x_scaled_train = StandardScaler().fit_transform(x_train)
x_scaled_test = StandardScaler().fit_transform(x_test)

# 网格搜索法寻找C与gamma最优值
grid = GridSearchCV(SVC(), param_grid={"C": [0.01, 0.1, 1, 10, 100],
                                       "gamma": [0.1, 0.2, 0.4, 0.8, 1.6]}, cv=3, scoring='accuracy', iid=True,
                    )
grid.fit(x_scaled_train, y_train)
print("最优参数为 %s " % grid.best_params_)

# 利用最优参数，核函数选择rbf，训练分类器
model = SVC(C=1, kernel='rbf', gamma=0.1)
model.fit(x_scaled_train, y_train)

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

# 输出结果
print('训练集的精度为：{:.4f}'.format(model.score(x_scaled_train, y_train)))
print('测试集的精度为：{:.4f}'.format(model.score(x_scaled_test, y_test)))