import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from itertools import combinations

# import training and holdout data frames from pkl files (that were created from running exploratory_analysis.py)
df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

"""
Returns separated data frame that contains feature data and response data 
Params:
    df: data frame to split based on features and response columns
Returns:
    x: data frame only containing features columns
    y: data frame only containing response column
"""
def separateXandY(df):
    features = list(filter(lambda k: ('label' not in k and 'img' not in k and 'lungStatus' not in k), df.columns))
    x = df[features]
    y = df['label']
    return x, y

"""
Returns calculated predictions and predicted probabilities
Params:
    model: model fitted using training set to be used to make predictions
    x_test: holdout set's explanatory data to make predictions on
Returns:
    pred: array of predicted classes ("Healthy", "Bacteria", "COVID-19", "Other Virus")
    pred_prob: array of predicted probabilities 
"""
def calculatePredict(model, x_test):
    pred = model.predict(x_test)
    pred_prob = model.predict_proba(x_test)
    return pred, pred_prob

"""
Returns selected features using feature selection based on importance
Params:
    model: model to find selected features for 
        if model == Logit Regression, selects based on feature coefficients
        if model == Ada Boost or Random Forest, selects based on importance 
    x_train: data frame containing training set's explanatory data
    y_train: data frame containing training set's response data 
Returns:
    selected_features: array of feature names selected to be used for model with subset of features 
"""
def findModelFeatures(model, x_train, y_train):
    features = x_train.columns.to_list()
    
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()

    model_select = SelectFromModel(estimator=model).fit(x_train, y_train)
    is_selected = model_select.get_support()

    selected_features = []
    for index, feature in enumerate(features):
        if is_selected[index]:
            selected_features.append(feature)
    print("- selected subset: " + str(selected_features))
    return selected_features

"""
Returns fitted model using subset of features 
Params:
    model: model to fit using the subset of features
    x_train: data frame containing training set's explanatory data
    y_train: data frame containing training set's response data 
Returns:
    fit_subset: fitted model using subset of features 
    selected_features: array of feature names selected to be used for model
"""
def trainSubsetModel(model, x_train, y_train):
    # performs feature selection for model
    selected_features = findModelFeatures(model, x_train, y_train)
    x_train_subset = x_train[selected_features]

    x_train_subset = x_train_subset.to_numpy()
    y_train = y_train.to_numpy().ravel()
    
    fit_subset = model.fit(x_train_subset, y_train)
    return fit_subset, selected_features

"""
Calculates and prints performance measures of fitted model
Params:
    actual: response from holdout set 
    preds: predicted classes made by fitted model
    probas: predicted probabilities made by fitted model
"""
def calculatePerformance(actual, preds, probas):
    # calculates point accuracy and F1 scores
    print('Overall accuracy: ', accuracy_score(actual, preds), '\n')
    f1_score = f1_scores(actual, preds)
    print('F1-Scores for Bacteria, COVID-19, Healthy, Other Virus\n', f1_score[2])
    print('Precision for Bacteria, COVID-19, Healthy, Other Virus\n', f1_score[0])
    print('Recall for Bacteria, COVID-19, Healthy, Other Virus\n', f1_score[1])
    print('Supports for Bacteria, COVID-19, Healthy, Other Virus\n', f1_score[3], '\n')

    # creates confusion matrix containing misclassification
    classes = actual.unique()
    cfmatrix = np.array(confusion_matrix(actual, preds, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))

    # calculates avg length, misclass rates, coverage rates
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


"""
Returns model with best AUC using hyperparameters found via grid search 4-fold cross validation 
Params:
    model: model to hypertune 
    params_grid: set of hyperparameters to test 
    x_train: data frame containing training set's explanatory data
    y_train: data frame containing training set's response data 
Returns:
    fitted model with hyperparameters that give best AUC 
"""
def hyperparamTuning(model, params_grid, x_train, y_train):
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    
    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=4, scoring='roc_auc_ovr', refit=True)
    grid_result = grid_search.fit(x_train, y_train)

    print("Best AUC: " + str(grid_result.best_score_) + " using " + str(grid_result.best_params_))
    return grid_result.best_estimator_

"""
Returns calculated 50% and 80% categorical prediction intervals 
Params:
    prob_matrix: matrix containing predicted probabilities of each class for each holdout case
    labels: array of possible classes ("Healthy", "Bacteria", "COVID-19", "Other Virus")
Returns:
    pred50: 50% prediction interval
    pred80: 80% prediction interval 
"""
def categoryPredInterval(prob_matrix, labels):
    n, k = prob_matrix.shape
    pred50 = arr_str = [''] * n
    pred80 = arr_str = [''] * n

    for i in range(n):
        p = prob_matrix[i,]
        ip = np.argsort(p)
        p_ordered = np.sort(p)
        labels_ordered = np.flip(labels[ip])
        G = np.flip(np.cumsum(np.insert(p_ordered, 0, 0)))
        k1 = np.min(np.where(G <= 0.5)[0])
        k2 = np.min(np.where(G <= 0.2)[0])

        pred1 = labels_ordered[0:k1]
        pred2 = labels_ordered[0:k2]

        pred50[i] = '.'.join(pred1)
        pred80[i] = '.'.join(pred2)

    return pred50, pred80

"""
Returns 50% or 80% confusion matrix 
Params:
    actual: true values of classes in holdout set
    pred: predicted classes for cases in holdout set 
Returns:
    confusion matrix 
"""
def contingencyMatrix(actual, pred):
    return pd.DataFrame(pd.crosstab(actual, pred), index=['Bacteria', 'COVID-19', 'Healthy', 'Other Virus'])

"""
Returns calculated performance measures based on confusion matrix 
Params:
    table: confusion matrix containing true values and predictions intervals of holdout set 
Returns:
    avg_len: average length of prediction interval
    miss: number of misclassified cases in prediction interval 
    miss_rate: misclassification rate of prediction interval 
    cover_rate: coverage rate of prediction interval 
"""
def coverage(table):
    nclass, nsubset = table.shape
    row_freq = table.sum(axis=1)
    labels = table.index
    subset_labels = table.columns
    cover = np.zeros(nclass)
    avg_len = np.zeros(nclass)

    for irow in range(nclass):
        for icol in range(nsubset):
            intervalSize = subset_labels[icol].count('.') + 1
            isCovered = subset_labels[icol].count(labels[irow]) == 1
            frequency = table[subset_labels[icol]].values[irow]
            cover[irow] = cover[irow] + frequency*isCovered
            avg_len[irow] = avg_len[irow] + frequency*intervalSize

    miss = row_freq - cover
    avg_len = avg_len / row_freq
    return avg_len, miss, miss/row_freq, cover/row_freq

"""
Plots comparison probability plot of the predictions of 2 fitted models
Params:
    method1: 1st method to compare
    method2: 2nd method to compare
    pred1: predicted probabilities of method1
    pred2: predicted probabilities of method2 
Returns:
    plot with method1 on x-axis, and method2 on y-axis, and values of scatterplot are predicted probabilities
"""
def comparisonProbabilityPlot(method1, method2, pred1, pred2):
    plt.scatter(np.asarray(pred1), np.asarray(pred2))
    plt.title("Probability Comparison plot")
    plt.xlabel(method1)
    plt.ylabel(method2)
    plt.savefig('images/compare/compare_' + method1 + '_' + method2)
    plt.clf()

"""
Returns calculated F1-score 
Params:
    actual: true values of classes in holdout set
    pred: predicted classes for cases in holdout set 
Returns:
   f1_score for fitted model
"""
def f1_scores(actual, pred):
    return precision_recall_fscore_support(actual, pred)


########################################################################################################################


"""
Fits, predicts and compares various methods of multinomial classification via out-of-sample measures. 
Given training with image augmentation, each method is hyperparameter tuned with balanced weights. 
For each method, 2 models are fitted. 1 uses all features while another use a subset of features found. 
Methods include:
1. Logistic Regression
2. Random Forest
3. AdaBoost
"""
def main():
    print("Perform Analysis:")
    x_train, y_train = separateXandY(df_train)
    x_test, y_test = separateXandY(df_test)

    test_probs = {}

    print("\n=============== LOGIT REGRESSION ===============")
    logit_grid = {
        'penalty': ['l2'],
        'C': [1.0, 10]
    }
    best_logit = hyperparamTuning(LogisticRegression(multi_class='multinomial', solver='saga', max_iter = 1000000, class_weight = 'balanced'),
                                  logit_grid, x_train, y_train)
    test_pred_logit, test_prob_logit = calculatePredict(best_logit, x_test)
    calculatePerformance(y_test, test_pred_logit, test_prob_logit)

    print("\n=============== LOGIT REGRESSION SUBSET ===============")
    best_logit_subset, selected_features_logit = trainSubsetModel(best_logit, x_train, y_train)
    test_pred_subset_logit, test_prob_subset_logit = calculatePredict(best_logit_subset, x_test[selected_features_logit])
    calculatePerformance(y_test, test_pred_subset_logit, test_prob_subset_logit)

    test_probs["Logit"] = test_prob_logit
    test_probs["Logit Subset"] = test_prob_subset_logit


    print("\n=============== RANDOM FOREST ===============")
    rf_grid = {
        'n_estimators': [200, 400, 1000, 1400],
        'max_depth':[10, 20, 30, None]
    }
    best_rf = hyperparamTuning(RandomForestClassifier(class_weight = 'balanced'),
                               rf_grid, x_train, y_train)
    test_pred_rf, test_prob_rf = calculatePredict(best_rf, x_test)
    calculatePerformance(y_test, test_pred_rf, test_prob_rf)

    print("\n=============== RANDOM FOREST SUBSET ===============")
    best_rf_subset, selected_features_rf = trainSubsetModel(best_rf, x_train, y_train)
    test_pred_subset_rf, test_prob_subset_rf = calculatePredict(best_rf_subset, x_test[selected_features_rf])
    calculatePerformance(y_test, test_pred_subset_rf, test_prob_subset_rf)

    test_probs["Random Forest"] = test_prob_rf
    test_probs["Random Forest Subset"] = test_prob_subset_rf


    print("\n=============== ADA BOOST ===============")
    ada_grid = {
        'n_estimators': [50,100,500,1000]
    }
    best_ada = hyperparamTuning(AdaBoostClassifier(),
                                ada_grid, x_train, y_train)
    test_pred_ada, test_prob_ada = calculatePredict(best_ada, x_test)
    calculatePerformance(y_test, test_pred_ada, test_prob_ada)

    print("\n=============== ADA BOOST SUBSET ===============")
    best_ada_subset, selected_features_ada = trainSubsetModel(best_ada, x_train, y_train)
    test_pred_subset_ada, test_prob_subset_ada = calculatePredict(best_ada_subset, x_test[selected_features_ada])
    calculatePerformance(y_test, test_pred_subset_ada, test_prob_subset_ada)

    test_probs["Ada Boost"] = test_prob_ada
    test_probs["Ada Boost Subset"] = test_prob_subset_ada

    # for each method, create comparison plots probabilities vs other method
    comparisons = combinations(test_probs.keys(), 2)
    for comparison in comparisons:
        # print(comparison)
        method1, method2 = comparison
        pred1 = test_probs[method1]
        pred2 = test_probs[method2]
        comparisonProbabilityPlot(method1, method2, pred1, pred2)

if __name__ == "__main__":
    main()