import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
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

    classes = y_train['label4'].unique()

    x_train_np = x_train.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    fit = model.fit(x_train_np, y_train)
    pred = fit.predict(x_test)
    probas = fit.predict_proba(x_test)
    
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
    
    
print("Perform Analysis")
print(" --> Note: used oversampling - used proportion of difference between max and covid count\n")
df_oversampled_train = oversampleCovid(df_train)

print("\nLogit Regression")
testPredLogit, testPredSubsetLogit = fitPredictModel(LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000), 
                                                     df_oversampled_train,
                                                     df_test)
probasLogit50, probasLogit80 = categoryPredInterval(testPredLogit, df_oversampled_train['label4'].unique())
probasLogitSub50, probasLogitSub80 = categoryPredInterval(testPredSubsetLogit, df_oversampled_train['label4'].unique())

print("\nRandom Forest")
testPredRf, testPredSubsetRf = fitPredictModel(RandomForestClassifier(n_estimators=1400, max_depth=220, max_features='auto'),
                                               df_oversampled_train,
                                               df_test)
probasRf50, probasRf80 = categoryPredInterval(testPredRf, df_oversampled_train['label4'].unique())
probasRfSub50, probasRfSub80 = categoryPredInterval(testPredSubsetRf, df_oversampled_train['label4'].unique())

print("\nAda Boost")
testPredAda, testPredSubsetAda = fitPredictModel(AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=1400, max_depth=220, max_features='auto')),
                                                 df_oversampled_train,
                                                 df_test)
probasAda50, probasAda80 = categoryPredInterval(testPredAda, df_oversampled_train['label4'].unique())
probasAdaSub50, probasAdaSub80 = categoryPredInterval(testPredSubsetAda, df_oversampled_train['label4'].unique())