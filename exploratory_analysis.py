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
    print ('Creation of the directory failed or already created')
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


print('Creating training set...')
df_transform = pd.DataFrame()

# middle of grayscale, 127.5
medianPixel = round(255 / 2)

for index, row in df_train.iterrows():
    img = cv2.imread('data/train/' + row['X_ray_image_name'], cv2.IMREAD_GRAYSCALE)
    imgr = cv2.resize(img, (200,200))

    # for calculation of stats
    unique, counts = np.unique(imgr, return_counts=True)
    pixelCounts = dict(zip(unique, counts))

    imgPixels = imgr.ravel()

    # light if > median; dark if < median
    lightPixels = imgPixels[imgPixels > medianPixel]
    darkPixels = imgPixels[imgPixels < medianPixel]
    
    numMedian = 0
    if medianPixel in pixelCounts.keys():
        numMedian = pixelCounts[medianPixel]

    df_transform = df_transform.append({
        'img': imgr,
        'label1': row['Label'],
        'label2': row['Label_1_Virus_category'],
        'label3': row['Label_2_Virus_category'],
        'avgBrightness': np.mean(imgr),
        'lightestPixel': np.max(imgr),
        'numOfLightest': pixelCounts[np.max(imgr)],
        'darkestPixel': np.min(imgr),
        'numOfDarkest': pixelCounts[np.min(imgr)],
        'numOfMedian': numMedian,
        'numAboveMedian': len(lightPixels),
        'numBelowMedian': len(darkPixels),
        'avgAboveMedian': np.mean(lightPixels),
        'avgBelowMedian': np.mean(darkPixels)
    }, ignore_index = True)
print('Finished creating training set...')


print('Creating histograms...')
countNormal = 0
countPenu = 0

avgCountsNormal = [0] * 256
avgCountsPneu = [0] * 256
for index, row in df_transform.iterrows():
    counts, bins = np.histogram(df_transform.iloc[index]['img'].ravel(), 256, (0,255))
    patientType = row['label1']
    if patientType == "Normal":
        avgCountsNormal = avgCountsNormal + counts
        countNormal += 1
    elif patientType == "Pnemonia":
        avgCountsPneu = avgCountsPneu + counts
        countPenu += 1

avgCountsNormal = avgCountsNormal / countNormal
avgCountsPneu = avgCountsPneu / countPenu

plt.bar(np.arange(0,256), avgCountsNormal)
plt.xlabel('Pixel Brightness')
plt.ylabel('Avg Count')
plt.savefig('images/hist_normal')

plt.clf()

plt.bar(np.arange(0,256), avgCountsPneu)
plt.xlabel('Pixel Brightness')
plt.ylabel('Avg Count')
plt.savefig('images/hist_pmeumonia')
print('Finished creating histograms...')

plt.clf()

print('Creating boxplots...')
# for every explanatory variable, create boxplot
for col in df_transform.columns:
    if col not in ('img', 'label1', 'label2', 'label3'):
        boxplot = df_transform.boxplot(by = 'label1', column = [col], grid = False)
        plt.title(col)
        plt.suptitle("")
        plt.savefig('images/boxplot_' + col)
print('Finished creating boxplots...')