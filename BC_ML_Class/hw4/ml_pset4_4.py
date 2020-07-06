from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

wine = load_wine()
numSamples = len(wine.data)

data = wine.data
target = wine.target

#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state = 0)




tuned_parameters = [{'kernel': ['poly'], 'C': [0.5, 1, 2], 'degree': [1,2,4]}]
scores = ['precision', 'recall']


for score in scores:
    clf = GridSearchCV(SVC(), tuned_parameters, iid = False)#34
    clf.fit(data, target)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = target, clf.predict(data)
    print(classification_report(y_true, y_pred))
    print()