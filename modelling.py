import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

# separates explanatory and response into 2 dataframes
def separateXandY(df):
    features = list(filter(lambda k: ('label' not in k and 'img' not in k), df.columns))
    x = df[features]
    y = df[['label4']]
    return x, y

# duplicates covid 19 rows in given training dataset k times
def oversampleCovid(df_train, k):
    classes = df_train['label4'].unique()
    for label in classes:
        print(label + " has num of rows: " + str(len(df_train.loc[df_train['label4'] == label])))

    # oversample covid-19 (covid has 50, while healthy and other vir has ~1345 and bacteria as 2530)
    covid_df = df_train.loc[df_train['label4'] == "COVID-19"]

    frames = [df_train]
    # duplicate covid rows k times
    for i in range(1, k):
        frames.append(covid_df)
    df_oversampled_train = pd.concat(frames)
    covid_df = df_oversampled_train.loc[df_train['label4'] == "COVID-19"]
    print("--> There are now " + str(len(covid_df)) + " rows with covid-19 labels. (done oversampling).")
    return df_oversampled_train

def fitLogitReg(df_train, df_test):
    df_oversampled_train = oversampleCovid(df_train, 6)
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

# multinomial logistic regression with all features
fitLogitReg(df_train, df_test)