from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()

data = wine.data
target = wine.target


x = 2**(0.5*np.arange(-8,9))
print(x)
accuracy = []
for i in range(len(x)):
    clf = LogisticRegression(solver = 'liblinear', C = x[i])
    clf.fit(data, target)
    score = cross_val_score(clf, data, target, cv=4)
    accuracy.append(np.mean(score))



plt.xlabel('Regularization constant')
plt.ylabel('accuracy')
plt.plot(x,accuracy)
#plt.yscale('log', basey=2)
plt.legend()
plt.show()



