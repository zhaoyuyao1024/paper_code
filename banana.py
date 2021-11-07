import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import pearsonr

features = []
dataMat = []
labelMat = []

# 导入数据
f = open('diabetes.txt')
# features存取特征名
features = f.readline()
# 逐行读取数据，dataMat存储特征数据，labelMat存储标签数据
for line in f.readlines():
    lineArr = line.strip().split(',')
    dataMat.append([int(lineArr[0]), int(lineArr[1]), int(lineArr[2]), int(lineArr[3]), int(lineArr[4]),
                    float(lineArr[5]),  float(lineArr[6]), int(lineArr[7])])
    labelMat.append(int(lineArr[8]))
f.close()

# 划分训练集与测试集，训练集为前60%，测试集为后40%
x_train = dataMat[:461]
x_test = dataMat[461:]
y_train = labelMat[:461]
y_test = labelMat[461:]

# 通过特征选择筛选出k个最佳特征
# print(SelectKBest(chi2, k=6).fit_transform(x_train, y_train)[0])

# 对数据进行归一化处理
x_scaled_train = StandardScaler().fit_transform(x_train)
x_scaled_test = StandardScaler().fit_transform(x_test)

# 网格搜索法寻找C与gamma最优值
grid = GridSearchCV(SVC(), param_grid={"C": [0.01, 0.1, 1.0, 10.0, 100.0],  "gamma": [0.1, 0.2, 0.4, 0.8, 1.6]},
                    cv=3, scoring='accuracy', iid=True)
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