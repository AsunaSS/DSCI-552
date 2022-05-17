from sklearn import svm
import numpy as np
X = [[1,2],[2,1],[0,0]]
x = np.array(X)
print(x)
y = [1, 1, -1]
model = svm.SVC(kernel='linear')
model = model.fit(x,y)
print(model.support_vectors_)
print(model.support_)
w = model.coef_[0]
print("w:",w)
a = -w[0] / w[1] # 斜率
xx = np.linspace(-5, 5) # 在区间[-5, 5] 中产生连续的值，用于画线
yy = a * xx - (model.intercept_[0]) / w[1]
b = model.support_vectors_[0] # 第一个分类的支持向量
print("b1:",b)
yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1] # 第二个分类中的支持向量
print("b2:",b)
yy_up = a * xx + (b[1] - a * b[0])
new_one = [[1,1]]
print(model.predict(new_one))

import pylab as pl

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(x[:, 0], x[:, 1], c=y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()