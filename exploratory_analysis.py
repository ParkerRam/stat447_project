import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import load_img

summary = pd.read_csv('data/Chest_xray_Corona_dataset_Summary.csv', index_col=0)
df = pd.read_csv('data/Chest_xray_Corona_Metadata.csv', index_col=0)

# Remove Stress-Smoking, SARS, Strep rows since there are so little of them
df = df[(df.Label_1_Virus_category != 'Stress-Smoking') &
        (df.Label_2_Virus_category != 'SARS') & 
        (df.Label_2_Virus_category != 'Streptococcus')]

# Replace NaNs with 'Other'
df = df.replace(np.nan, 'Other', regex=True)

# Move 8 COVID-19 cases to test set since it has none of them
count = 0
for index, row in df.iterrows():
    if (row['Label_2_Virus_category'] == 'COVID-19') & (row['Dataset_type'] == 'TRAIN'):
        row['Dataset_type'] = 'TEST'
        count += 1
    if count > 7:
        break
        
df_train = df[df['Dataset_type'] == 'TRAIN']
df_test = df[df['Dataset_type'] == 'TEST']

# Create images directory
try:
    os.mkdir('images/')
except OSError:
    print ('Creation of the directory failed')
else:
    print ('Successfully created the directory')

# Example x-ray of normal lung
img_name = df_train.loc[df_train['Label'] == 'Normal'].iloc[0]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/normal.png')
print('Image of normal lung saved to images/')

# Example x-ray of virus lung
img_name = df_train.loc[(df_train['Label'] == 'Pnemonia') & 
                       (df_train['Label_1_Virus_category'] == 'Virus')].iloc[0]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/virus.png')
print('Image of virus lung saved to images/')

# Example x-ray of bacteria lung
img_name = df_train.loc[(df_train['Label'] == 'Pnemonia') & 
                       (df_train['Label_1_Virus_category'] == 'bacteria')].iloc[0]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/bacteria.png')
print('Image of bacteria lung saved to images/')


# Setting data up
# We may want to drop some rows (ARDS, combine Strep), set this up as a function to use for train and test, split this into new script later
df_transform = pd.DataFrame(columns = ['img', 'label1', 'label2', 'label3'])

print('Creating training set...')
for index, row in df_train.iterrows():
    img = cv2.imread('data/train/' + row['X_ray_image_name'], cv2.IMREAD_GRAYSCALE)
    imgr = cv2.resize(img, (100,100))
    df_transform = df_transform.append({
        'img': imgr,
        'label1': row['Label'],
        'label2': row['Label_1_Virus_category'],
        'label3': row['Label_2_Virus_category'],
    }, ignore_index = True)
print('Finished creating training set...')