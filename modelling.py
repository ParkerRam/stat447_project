import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import combinations

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

# separates explanatory and response into 2 dataframes
def separateXandY(df):
    features = list(filter(lambda k: ('label' not in k and 'img' not in k and 'lung_status' not in k), df.columns))
    x = df[features]
    y = df[['label']]
    return x, y

def findModelFeatures(model, x_train, y_train):
    features = x_train.columns.to_list()
    x_train = x_train.to_numpy()

    model_select = SelectFromModel(estimator=model).fit(x_train, y_train)
    is_selected = model_select.get_support()

    selectedFeatures = []
    for index, feature in enumerate(features):
        if is_selected[index]:
            selectedFeatures.append(feature)
    print("- selected subset: " + str(selectedFeatures))
    return selectedFeatures

def fitPredictModel(model, df_train, df_test):
    x_train, y_train = separateXandY(df_train)
    x_test, y_test = separateXandY(df_test)

    classes = y_train['label'].unique()

    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    pred = model.predict(x_test)
    probas = model.predict_proba(x_test)

    cfmatrix = np.array(confusion_matrix(y_test, pred, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))

    print("\nModel using subset of features")
    selectedFeatures = findModelFeatures(model, x_train, y_train)
    x_train_subset = x_train[selectedFeatures]
    x_test_subset = x_test[selectedFeatures]

    fit_subset = model.fit(x_train_subset.to_numpy(), y_train)
    pred_subset = fit_subset.predict(x_test_subset)
    probas_subset = fit_subset.predict_proba(x_test_subset)

    cfmatrix_subset = np.array(confusion_matrix(y_test, pred_subset, labels=classes))
    print(pd.DataFrame(cfmatrix_subset, index=classes, columns=classes))

    return probas, probas_subset

def hyperparamTuning(model, params_grid, df_train, df_test):
    x_train, y_train = separateXandY(df_train)
    x_test, y_test = separateXandY(df_test)

    classes = y_train['label'].unique()

    x_train_np = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=4, scoring='roc_auc_ovr', refit=True)
    grid_result = grid_search.fit(x_train_np, y_train)

    print("Best: " + str(grid_result.best_score_) + " using " + str(grid_result.best_params_))
    
    return grid_result.best_estimator_

def categoryPredInterval(probMatrix, labels):
    n, k = probMatrix.shape
    pred50 = arr_str = [''] * n
    pred80 = arr_str = [''] * n

    for i in range(n):
        p = probMatrix[i,]
        ip = np.argsort(p)
        pOrdered = np.sort(p)
        labelsOrdered = np.flip(labels[ip])
        G = np.flip(np.cumsum(np.insert(pOrdered, 0, 0)))
        k1 = np.min(np.where(G <= 0.5)[0])
        k2 = np.min(np.where(G <= 0.2)[0])

        pred1 = labelsOrdered[0:k1]
        pred2 = labelsOrdered[0:k2]

        pred50[i] = '.'.join(pred1)
        pred80[i] = '.'.join(pred2)

    return pred50, pred80

def contingencyMatrix(actual, pred):
    return pd.DataFrame(pd.crosstab(actual, pred), index=['Bacteria', 'COVID-19', 'Healthy', 'Other Virus'])

def coverage(table):
    nclass, nsubset = table.shape
    rowFreq = table.sum(axis=1)
    labels = table.index
    subsetLabels = table.columns
    cover = np.zeros(nclass)
    avgLen = np.zeros(nclass)

    for irow in range(nclass):
        for icol in range(nsubset):
            intervalSize = subsetLabels[icol].count('.') + 1
            isCovered = subsetLabels[icol].count(labels[irow]) == 1
            frequency = table[subsetLabels[icol]].values[irow]
            cover[irow] = cover[irow] + frequency*isCovered
            avgLen[irow] = avgLen[irow] + frequency*intervalSize

    miss = rowFreq - cover
    avgLen = avgLen / rowFreq
    return avgLen, miss, miss/rowFreq, cover/rowFreq

def comparisonProbabilityPlot(method1, method2, pred1, pred2):
    plt.scatter(np.asarray(pred1), np.asarray(pred2))
    plt.title("Probability Comparison plot")
    plt.xlabel(method1)
    plt.ylabel(method2)
    plt.savefig('images/compare/compare_' + method1 + '_' + method2)
    plt.clf()


print("Perform Analysis:")

testPreds = {}

# print("\nLogit Regression")
# best_logit = hyperparamTuning(LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 1000000, class_weight = 'balanced'),
#                               {
#                                   'penalty': ['l2'],
#                                   'C': [1.0]
#                               },
#                               df_train,
#                               df_test)
# testPredLogit, testPredSubsetLogit = fitPredictModel(best_logit,
#                                                      df_train,
#                                                      df_test)
# testPreds["Logit"] = testPredLogit
# testPreds["Logit Subset"] = testPredSubsetLogit


print("\nRandom Forest")
best_rf = hyperparamTuning(RandomForestClassifier(class_weight = 'balanced'),
                          {
                              'n_estimators': [100, 500, 1000, 1500],
                              'max_depth': [1, 10, None]
                          },
                          df_train,
                          df_test)
testPredRf, testPredSubsetRf = fitPredictModel(best_rf,
                                               df_train,
                                               df_test)
testPreds["Random Forest"] = testPredRf
testPreds["Random Forest Subset"] = testPredSubsetRf


print("\nAda Boost")
best_ada = hyperparamTuning(AdaBoostClassifier(),
                           {
                               'n_estimators': [50,100,500,1000]
                           },
                           df_train,
                           df_test)
testPredAda, testPredSubsetAda = fitPredictModel(best_ada,
                                                 df_train,
                                                 df_test)
testPreds["Ada Boost"] = testPredAda
testPreds["Ada Boost Subset"] = testPredSubsetAda

# for each method, create comparison plots probabilities vs other method
comparisons = combinations(testPreds.keys(), 2)
for comparison in comparisons:
    # print(comparison)
    method1, method2 = comparison
    pred1 = testPreds[method1]
    pred2 = testPreds[method2]
    comparisonProbabilityPlot(method1, method2, pred1, pred2)

# calculate coverage scores for each method
for method in testPreds.keys():
    print("Scores for " + method)
    testPred = testPreds[method]
    probs50, probs80 = categoryPredInterval(testPred, np.asarray(['Bacteria', 'COVID-19', 'Healthy', 'Other Virus']))
    scores50 = coverage(contingencyMatrix(df_test['label'], np.asarray(probs50)))
    scores80 = coverage(contingencyMatrix(df_test['label'], np.asarray(probs80)))
    print(" - scores for 50% pred intervals:")
    print(" --> Avg length:\n" + str(scores50[0]))
    print(" --> Misclass:\n" + str(scores50[1]))
    print(" --> Misclass Rate:\n" + str(scores50[2]))
    print(" --> Coverage Rate:\n" + str(scores50[3]))

    print("\n - scores for 80% pred intervals:")
    print(" --> Avg length:\n" + str(scores80[0]))
    print(" --> Misclass:\n" + str(scores80[1]))
    print(" --> Misclass Rate:\n" + str(scores80[2]))
    print(" --> Coverage Rate:\n" + str(scores80[3]))
    print("\n")
