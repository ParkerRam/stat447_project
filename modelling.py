import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

def separateXandY(df):
    # extracts features
    features = list(filter(lambda k: ('label' not in k and 'img' not in k and 'img_name' not in k), df.columns))
    x = df[features]
    y = df[['label4']]
    return x, y

def oversampleCovid(df_train):
    classes = df_train['label4'].unique()
    for label in classes:
        print(label + " has num of rows: " + str(len(df_train.loc[df_train['label4'] == label])))

    # oversample covid-19 (covid has 50, while healthy and other vir has ~1345 and bacteria as 2530)
    # will oversample 6x time
    covid_df = df_train.loc[df_train['label4'] == "COVID-19"]

    frames = [df_train]
    for i in range(1, 6):
        frames.append(covid_df)
    df_oversampled_train = pd.concat(frames)
    print("--> There are now " + str(len(df_train)) + " rows with covid-19 labels. (done oversampling).")

    print(len(df_oversampled_train.loc[df_oversampled_train['label4'] == "COVID-19"]))
    return df_oversampled_train

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

    fit1 = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(x_train, y_train)
    pred1 = fit1.predict(x_test)
    cfmatrix1 = np.array(confusion_matrix(y_test, pred1, labels=classes))
    print(pd.DataFrame(cfmatrix1, index=classes, columns=classes))

    # fit2 = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=8000).fit(x_train, y_train)
    # pred2 = fit2.predict(x_test)
    # cfmatrix2 = np.array(confusion_matrix(y_test, pred2, labels=classes))
    # print(pd.DataFrame(cfmatrix2, index=classes, columns=classes))


# multinomial logistic regression with all features
fitLogitReg(df_train, df_test)