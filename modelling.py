import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

# separates explanatory and response into 2 dataframes
def separateXandY(df):
    features = list(filter(lambda k: ('label' not in k and 'img' not in k), df.columns))
    x = df[features]
    y = df[['label4']]
    return x, y

# duplicates covid 19 rows in given training dataset based on max count for labels
# covid has 50 samples, and maxCount = 2350 (bacteria)
def oversampleCovid(df_train):
    classes = df_train['label4'].unique()
    covid_df = df_train.loc[df_train['label4'] == "COVID-19"]
    covidCount = len(covid_df)
    maxCount = 0

    for label in classes:
        count = len(df_train.loc[df_train['label4'] == label])
        print(label + " has num of rows: " + str(count))
        if count > maxCount:
            maxCount = count
    k = math.floor((maxCount / covidCount))

    print("\nOversampling by duplicating " + str(k) + " times: ")
    frames = [df_train]
    # duplicate covid rows k times
    for i in range(1, k):
        frames.append(covid_df)
    df_oversampled_train = pd.concat(frames)
    covid_df = df_oversampled_train.loc[df_oversampled_train['label4'] == "COVID-19"]
    print("-> There are now " + str(len(covid_df)) + " rows with covid-19 labels. (done oversampling).")
    return df_oversampled_train

# this chose vars: aboveMedianVar, belowMedianVar, numAboveMedian, numDarkest, numMedian, shadeVar, x2bar
def findLogitFeatures(model, x_train, y_train):
    features = x_train.columns.to_list()
    x_train = x_train.to_numpy()
    logit_select = SelectFromModel(estimator=model).fit(x_train, y_train)
    is_selected = logit_select.get_support()
    selectedFeatures = []
    for index, feature in enumerate(features):
        if is_selected[index]:
            selectedFeatures.append(feature)
    print("- selected subset: " + str(selectedFeatures))
    return selectedFeatures

# fit logit model with oversampling and predict
def fitLogitReg(oversampled_train, df_test):
    x_train, y_train = separateXandY(oversampled_train)
    x_test, y_test = separateXandY(df_test)

    classes = y_train['label4'].unique()

    x_train_list = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy()

    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000)
    fit = model.fit(x_train_list, y_train)
    pred = fit.predict(x_test)
    cfmatrix = np.array(confusion_matrix(y_test, pred, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))

    print("\nLogit Reg subset")
    selectedFeatures = findLogitFeatures(model, x_train, y_train)
    x_train_subset = x_train[selectedFeatures]
    x_test_subset = x_test[selectedFeatures]
    fit_subset = model.fit(x_train_subset.to_numpy(), y_train)
    pred_subset = fit_subset.predict(x_test_subset)
    cfmatrix_subset = np.array(confusion_matrix(y_test, pred_subset, labels=classes))
    print(pd.DataFrame(cfmatrix_subset, index=classes, columns=classes))

def fitRandomForest(df_train, df_test):
    df_oversampled_train = oversampleCovid(df_train)
    train = separateXandY(df_oversampled_train)
    test = separateXandY(df_test)
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    y_test = test[1]

    classes = y_train['label4'].unique()

    y_train = y_train.values.ravel()

    # create random forest
    #rf = RandomForestClassifier()
    rf = RandomForestClassifier(n_estimators= 1400, max_depth= 220, max_features='auto')
    rf.fit(x_train, y_train)

    #predict
    rf_predict = rf.predict(x_test)

    # gives me warnings of ValueError: multiclass format is not supported
    #rf_cv_score = cross_val_score(rf, x_train, y_train, cv=10, scoring='roc_auc')

    rf_matrix = (confusion_matrix(y_test, rf_predict, labels=classes))

    print("Random Forest Confusion Matrix")
    print(pd.DataFrame(rf_matrix, index=classes, columns=classes))

# Find best subset through feature selection
# These were the features it picked:
# 'aboveMedianVar', 'belowMedianVar', 'numOfDarkest', 'numOfLightest',
# 'x2bar', 'x2ybr', 'xy2br'
def randomForestFeatures(df_train):
    df_oversampled_train = oversampleCovid(df_train)
    train = separateXandY(df_oversampled_train)
    x_train = train[0]
    y_train = train[1]

    y_train = y_train.values.ravel()

    rf_select = SelectFromModel(RandomForestClassifier(n_estimators=100))
    rf_select.fit(x_train, y_train)

    ## To see which features are important on fitted model
    rf_select.get_support()
    selected_feat= x_train.columns[(rf_select.get_support())]
    len(selected_feat)
    print(selected_feat)
    
def findRidgeFeatures(model, x_train, y_train):
    features = x_train.columns.to_list()
    x_train = x_train.to_numpy()
    ridge_select = SelectFromModel(estimator=model).fit(x_train, y_train)
    is_selected = ridge_select.get_support()
    selectedFeatures = []
    for index, feature in enumerate(features):
        if is_selected[index]:
            selectedFeatures.append(feature)
    print("- selected subset: " + str(selectedFeatures))
    return selectedFeatures
    
def fitRidgeClassifier(oversampled_train, df_test):
    x_train, y_train = separateXandY(oversampled_train)
    x_test, y_test = separateXandY(df_test)

    classes = y_train['label4'].unique()

    x_train_list = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy()

    model = RidgeClassifier(alpha=0.5)
    fit = model.fit(x_train_list, y_train)
    pred = fit.predict(x_test)
    cfmatrix = np.array(confusion_matrix(y_test, pred, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))

    print("\nRidge Classifier subset")
    selectedFeatures = findRidgeFeatures(model, x_train, y_train)
    x_train_subset = x_train[selectedFeatures]
    x_test_subset = x_test[selectedFeatures]
    fit_subset = model.fit(x_train_subset.to_numpy(), y_train)
    pred_subset = fit_subset.predict(x_test_subset)
    cfmatrix_subset = np.array(confusion_matrix(y_test, pred_subset, labels=classes))
    print(pd.DataFrame(cfmatrix_subset, index=classes, columns=classes))


print("Perform Analysis")
print(" --> Note: used oversampling - used proportion of difference between max and covid count\n")
df_oversampled_train = oversampleCovid(df_train)

# multinomial logistic regression with all features
print("\nLogit Regression")
fitLogitReg(df_oversampled_train, df_test)

print("\nRandom Forest")
fitRandomForest(df_train, df_test)

print("\nRidge Classifier")
fitRidgeClassifier(df_oversampled_train, df_test)

