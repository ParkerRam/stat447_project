import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

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

# fit logit model with oversampling and predict
def fitLogitReg(df_train, df_test):
    df_oversampled_train = oversampleCovid(df_train)
    train = separateXandY(df_oversampled_train)
    test = separateXandY(df_test)
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    y_test = test[1]

    classes = y_train['label4'].unique()
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy()

    fit = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000).fit(x_train, y_train)
    pred = fit.predict(x_test)
    cfmatrix = np.array(confusion_matrix(y_test, pred, labels=classes))
    print(pd.DataFrame(cfmatrix, index=classes, columns=classes))

def findLogitRegSubset(df_train):
    df_oversampled_train = oversampleCovid(df_train)
    train = separateXandY(df_oversampled_train)
    x_train = train[0]
    y_train = train[1]
    model = sm.mnlogit(endog=y_train, exog=x_train).fit()
    print(model.summary())

# multinomial logistic regression with all features
print("Logit with oversampling - used proportion of difference between max and covid count")
fitLogitReg(df_train, df_test)

# TODO: fit logit reg w/ subset of features
# subset = findLogitRegSubset(df_train)


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
    
fitRandomForest(df_train, df_test)


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

