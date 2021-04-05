import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from itertools import combinations

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

def separateXandY(df):
    features = list(filter(lambda k: ('label' not in k and 'img' not in k and 'lungStatus' not in k), df.columns))
    x = df[features]
    y = df['label']
    return x, y

def model_predict(model, x_test):
    return model.predict(x_test), model.predict_proba(x_test)

def findModelFeatures(model, x_train, y_train):
    features = x_train.columns.to_list()
    
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()

    model_select = SelectFromModel(estimator=model).fit(x_train, y_train)
    is_selected = model_select.get_support()

    selectedFeatures = []
    for index, feature in enumerate(features):
        if is_selected[index]:
            selectedFeatures.append(feature)
    print("- selected subset: " + str(selectedFeatures))
    return selectedFeatures

def train_subset_model(model, x_train, y_train):
    selectedFeatures = findModelFeatures(model, x_train, y_train)
    x_train_subset = x_train[selectedFeatures]
    
    x_train_subset = x_train_subset.to_numpy()
    y_train = y_train.to_numpy().ravel()
    
    fit_subset = model.fit(x_train_subset, y_train)
    return fit_subset, selectedFeatures

def fitPredictModel(actual, preds, probas):
    print('Overall accuracy: ', accuracy_score(actual, preds), '\n')
    f1score = f1_scores(actual, preds)
    print('F1-Scores for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[2])
    print('Precision for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[0])
    print('Recall for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[1])
    print('Supports for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[3], '\n')
    
    classes = actual.unique()
    cfmatrix = np.array(confusion_matrix(actual, preds, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))
    
    print('\nPrediction Interval Scores\n')
    probs50, probs80 = categoryPredInterval(probas, np.asarray(['Bacteria', 'COVID-19', 'Healthy', 'Other Virus']))
    scores50 = coverage(contingencyMatrix(actual, np.asarray(probs50)))
    scores80 = coverage(contingencyMatrix(actual, np.asarray(probs80)))
    print(' - scores for 50% pred intervals:')
    print(' --> Avg length:\n' + str(scores50[0]))
    print(' --> Misclass:\n' + str(scores50[1]))
    print(' --> Misclass Rate:\n' + str(scores50[2]))
    print(' --> Coverage Rate:\n' + str(scores50[3]))

    print('\n - scores for 80% pred intervals:')
    print(' --> Avg length:\n' + str(scores80[0]))
    print(' --> Misclass:\n' + str(scores80[1]))
    print(' --> Misclass Rate:\n' + str(scores80[2]))
    print(' --> Coverage Rate:\n' + str(scores80[3]))
    print('\n')

def hyperparamTuning(model, params_grid, x_train, y_train):
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    
    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=4, scoring='roc_auc_ovr', refit=True)
    grid_result = grid_search.fit(x_train, y_train)

    print("Best AUC: " + str(grid_result.best_score_) + " using " + str(grid_result.best_params_))
    
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

def f1_scores(actual, pred):
    return precision_recall_fscore_support(actual, pred)


########################################################################################################################

print("Perform Analysis:")
x_train, y_train = separateXandY(df_train)
x_test, y_test = separateXandY(df_test)

testPreds = {}

print("\n=============== LOGIT REGRESSION ===============")
logit_grid = {
    'penalty': ['l2'],
    'C': [1.0, 10]
}
best_logit = hyperparamTuning(LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 1000000, class_weight = 'balanced'),
                              logit_grid, x_train, y_train)
testPredLogit, testProbLogit = model_predict(best_logit, x_test)
fitPredictModel(y_test, testPredLogit, testProbLogit)

print("\n=============== LOGIT REGRESSION SUBSET ===============")
best_logit_subset, selected_features_logit = train_subset_model(best_logit, x_train, y_train)
testPredSubsetLogit, testProbSubsetLogit = model_predict(best_logit_subset, x_test[selected_features_logit])
fitPredictModel(y_test, testPredSubsetLogit, testProbSubsetLogit)

testPreds["Logit"] = testProbLogit
testPreds["Logit Subset"] = testProbSubsetLogit


print("\n=============== RANDOM FOREST ===============")
rf_grid = {
    'n_estimators': [200, 400, 1000, 1400],
    'max_depth':[10, 20, 30, None]
}
best_rf = hyperparamTuning(RandomForestClassifier(class_weight = 'balanced'),
                           rf_grid, x_train, y_train)
testPredRf, testProbRf = model_predict(best_rf, x_test)
fitPredictModel(y_test, testPredRf, testProbRf)

print("\n=============== RANDOM FOREST SUBSET ===============")
best_rf_subset, selected_features_rf = train_subset_model(best_rf, x_train, y_train)
testPredSubsetRf, testProbSubsetRf = model_predict(best_rf_subset, x_test[selected_features_rf])
fitPredictModel(y_test, testPredSubsetRf, testProbSubsetRf)

testPreds["Random Forest"] = testProbRf
testPreds["Random Forest Subset"] = testProbSubsetRf


print("\n=============== ADA BOOST ===============")
ada_grid = {
    'n_estimators': [50,100,500,1000]
}
best_ada = hyperparamTuning(AdaBoostClassifier(),
                           ada_grid, x_train, y_train)
testPredAda, testProbAda = model_predict(best_ada, x_test)
fitPredictModel(y_test, testPredAda, testProbAda)

print("\n=============== ADA BOOST SUBSET ===============")
best_ada_subset, selected_features_ada = train_subset_model(best_ada, x_train, y_train)
testPredSubsetAda, testProbSubsetAda = model_predict(best_ada_subset, x_test[selected_features_ada])
fitPredictModel(y_test, testPredSubsetAda, testProbSubsetAda)

testPreds["Ada Boost"] = testProbAda
testPreds["Ada Boost Subset"] = testProbSubsetAda


# for each method, create comparison plots probabilities vs other method
comparisons = combinations(testPreds.keys(), 2)
for comparison in comparisons:
    # print(comparison)
    method1, method2 = comparison
    pred1 = testPreds[method1]
    pred2 = testPreds[method2]
    comparisonProbabilityPlot(method1, method2, pred1, pred2)