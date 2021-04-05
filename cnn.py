import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, MaxPool2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train = pd.read_pickle('data/train_metadata.pkl')
df_test = pd.read_pickle('data/test_metadata.pkl')

# Split into training and validation sets
features = list(filter(lambda k: ('allLabel' not in k), train.columns))
X = train[features]
y = train['allLabel']
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

df_train = pd.DataFrame(np.column_stack((X_train, y_train)), columns=train.columns)
df_val = pd.DataFrame(np.column_stack((X_val, y_val)), columns=train.columns)

# Setup augmentation identical to before but including validation set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   featurewise_center=False,
                                   samplewise_center=False, 
                                   featurewise_std_normalization=False, 
                                   samplewise_std_normalization=False, 
                                   zca_whitening=False, 
                                   rotation_range = 30,  
                                   zoom_range = 0.2,
                                   width_shift_range=0.1,  
                                   height_shift_range=0.1,  
                                   horizontal_flip = True, 
                                   vertical_flip=False)
test_datagen = ImageDataGenerator(rescale = 1./255) 

training_set = train_datagen.flow_from_dataframe(dataframe = df_train,
                                                 directory = 'data/train',
                                                 x_col = 'X_ray_image_name',
                                                 y_col = 'allLabel', 
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical')

validation_set = test_datagen.flow_from_dataframe(dataframe = df_val,
                                                  directory = 'data/train',
                                                  x_col = 'X_ray_image_name',
                                                  y_col = 'allLabel', 
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  color_mode = 'grayscale',
                                                  class_mode = 'categorical')

test_set = test_datagen.flow_from_dataframe(dataframe = df_test,
                                            directory = 'data/test',
                                            x_col = 'X_ray_image_name',
                                            y_col = 'allLabel', 
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical',
                                            shuffle = False)

cnn = Sequential()

#Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)))

#Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu"))

# 2nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
cnn.add(Flatten())

# Fully Connected Layers
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'softmax', units = 4))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print('Training CNN...\n')
cnn_model = cnn.fit(training_set,
                    epochs = 160,
                    validation_data = validation_set)
print('Finished training CNN\n')

# Accuracy 
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('images/cnn_epochs')

def fitPredictModel(actual, preds, probas):
    print('Overall accuracy: ', accuracy_score(actual, preds), '\n')
    f1score = f1_scores(actual, preds)
    print('F1-Scores for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[2])
    print('Precision for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[0])
    print('Recall for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[1])
    print('Supports for Bacteria, COVID-19, Healthy, Other Virus\n', f1score[3], '\n')
    
    classes = np.unique(actual)
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
    
def f1_scores(actual, pred):
    return precision_recall_fscore_support(actual, pred)

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

# Get true labels in test set and convert from numerical to labels
actual = pd.get_dummies(pd.Series(test_set.classes)).to_numpy()
actual = actual.argmax(axis=-1)
actual = np.where(actual == 0, 'Bacteria', actual)
actual = np.where(actual == '1', 'COVID-19', actual)
actual = np.where(actual == '2', 'Healthy', actual)
actual = np.where(actual == '3', 'Other Virus', actual)

# Predict on test set
test_probs_cnn = cnn.predict(test_set)
cnn.evaluate(test_set)

# Convert numerized predictions to labels
test_preds_cnn = test_probs_cnn.argmax(axis=-1)
test_preds_cnn = np.where(test_preds_cnn == 0, 'Bacteria', test_preds_cnn)
test_preds_cnn = np.where(test_preds_cnn == '1', 'COVID-19', test_preds_cnn)
test_preds_cnn = np.where(test_preds_cnn == '2', 'Healthy', test_preds_cnn)
test_preds_cnn = np.where(test_preds_cnn == '3', 'Other Virus', test_preds_cnn)

# Evaluate CNN
print('CNN Results')
fitPredictModel(actual, test_preds_cnn, test_probs_cnn)