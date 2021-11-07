from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

np.random.seed(100)
# 导入数据
data = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
label = np.loadtxt('diabetes.txt', delimiter=',', skiprows=1, usecols=(8))
insulin = []
skin =[]
i = 0
for i in range(0, 768):
     if (data[:, 4][i] != 0):
        insulin.append(data[:, 4][i])
     if (data[:, 3][i] != 0):
        skin.append(data[:, 3][i])
insulin = np.sort(insulin)
skinthickness = np.sort(skin)

for i in range(768):
    if (data[:, 3][i] ==0):
        data[:, 3][i] = np.mean(skin)+np.random.randint(-10, 10, 1)
    if (data[:, 4][i]==0):
        data[:, 4][i] = np.mean(insulin)+np.random.randint(-30, 30, 1)

# 划分训练集与测试集，训练集为前60%，测试集为后40%
x_train = data[:461]
x_test = data[461:]
y_train = label[:461]
y_test = label[461:]

# 通过特征选择筛选出k个最佳特征
#print(SelectKBest(chi2, k=6).fit_transform(x_train, y_train)[0])

# 对数据进行归一化处理
x_scaled_train = preprocessing.StandardScaler().fit_transform(x_train)
x_scaled_test = preprocessing.StandardScaler().fit_transform(x_test)

train_accuracy = []
test_accuracy = []
k = range(1, 21)

for i in k:
    model = KNeighborsClassifier(i)
    model.fit(x_scaled_train, y_train)
    train_accuracy.append(model.score(x_scaled_train, y_train))
    test_accuracy.append(model.score(x_scaled_test, y_test))
plt.figure()
plt.title('Accuracy-K')
plt.xlabel('K neighbors')
plt.ylabel('Accuracy')
plt.plot(k, train_accuracy, label="train_accuracy")
plt.plot(k, test_accuracy, label="test_accuracy")
plt.legend()
plt.show()

model = KNeighborsClassifier(10, p=2)
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

print('训练集的精度为：{:.4f}'.format(model.score(x_scaled_train, y_train)))
print('测试集的精度为：{:.4f}'.format(model.score(x_scaled_test, y_test)))