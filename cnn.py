import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, MaxPool2D, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df_train = pd.read_pickle('data/train.pkl')
df_test = pd.read_pickle('data/test.pkl')

df_train = df_train[['img', 'label']]
df_test = df_test[['img', 'label']]

# Convert labels to numeric
clean_labels = {
    'Healthy': 1,
    'Bacteria': 2,
    'Other Virus': 3,
    'COVID-19': 4
}
df_train = df_train.replace(clean_labels)
df_test = df_test.replace(clean_labels)

# Scale pixels
df_train['img'] = df_train['img'].div(255)
df_test['img'] = df_test['img'].div(255)

# Convert to numpy arrays and reshape for cnn
X_train = np.stack(df_train['img'].values)
X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
y_train = np.stack(df_train['label'].values)


cnn = Sequential()

#Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(200, 200, 1)))

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
cnn.add(Dense(activation = 'sigmoid', units = 1))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn_model = cnn.fit(X_train,
                    y_train,
                    epochs = 30,
                    validation_split = 0.7,
                    validation_steps = 624)