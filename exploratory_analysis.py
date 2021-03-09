import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import load_img

summary = pd.read_csv('data/Chest_xray_Corona_dataset_Summary.csv', index_col=0)
df = pd.read_csv('data/Chest_xray_Corona_Metadata.csv', index_col=0)


################################ Cleaning data #####################################

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
        try:
            os.rename('data/train/' + row['X_ray_image_name'], 'data/test/' + row['X_ray_image_name'])
            print(row['X_ray_image_name'] + ' has been moved to TEST')
        except:
            print(row['X_ray_image_name'] + ' already moved')
        count += 1
    if count > 7:
        break
        
df_train = df[df['Dataset_type'] == 'TRAIN']
df_test = df[df['Dataset_type'] == 'TEST']

################################ Example images #####################################

# Create images directory
try:
    os.mkdir('images/')
except OSError:
    print ('Creation of the directory failed or already created')
else:
    print ('Successfully created the directory')

# Example x-ray of normal lung
img_name = df_train.loc[df_train['Label'] == 'Normal'].iloc[5]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/normal.png')
print('Image of normal lung saved to images/')

# Example x-ray of virus lung
img_name = df_train.loc[(df_train['Label'] == 'Pnemonia') & 
                       (df_train['Label_1_Virus_category'] == 'Virus')].iloc[3]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/virus.png')
print('Image of virus lung saved to images/')

# Example x-ray of bacteria lung
img_name = df_train.loc[(df_train['Label'] == 'Pnemonia') & 
                       (df_train['Label_1_Virus_category'] == 'bacteria')].iloc[10]['X_ray_image_name']
img_path = 'data/train/' + img_name
load_img(img_path).save('images/bacteria.png')
print('Image of bacteria lung saved to images/')

################################ Transform data #####################################
PIXELS_RESIZE = 200
medianPixel = round(255 / 2)

def transform_data(whichSet):
    print('Creating ' + whichSet + ' set...')
    
    df_transform = pd.DataFrame()
    
    df = df_train if whichSet == 'train' else df_test
    for index, row in df.iterrows():
        img = cv2.imread('data/' + whichSet + '/' + row['X_ray_image_name'], cv2.IMREAD_GRAYSCALE)
        try:
            imgr = cv2.resize(img, (PIXELS_RESIZE, PIXELS_RESIZE))
        except Exception as e:
            print('Removed ' + row['X_ray_image_name'] + ' since it is broken')
            continue

        # for calculation of stats
        unique, counts = np.unique(imgr, return_counts=True)
        pixelCounts = dict(zip(unique, counts))

        imgPixels = imgr.ravel()

        # light if > median; dark if < median
        lightPixels = imgPixels[imgPixels > medianPixel]
        darkPixels = imgPixels[imgPixels < medianPixel]
        
        xpos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
        ypos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
        for x in range(PIXELS_RESIZE):
            for y in range(PIXELS_RESIZE):
                xpos[x,y] = x*(imgr[x,y]/255)
                ypos[x,y] = y*(imgr[x,y]/255)

                
        numMedian = 0
        if medianPixel in pixelCounts.keys():
            numMedian = pixelCounts[medianPixel]

        df_transform = df_transform.append({
            'img': imgr,
            'label1': row['Label'],
            'label2': row['Label_1_Virus_category'],
            'label3': row['Label_2_Virus_category'],
            'shadeAvg': np.mean(imgr),
            'shadeVar': np.var(imgr),
            'lightestShade': np.max(imgr),
            'numOfLightest': pixelCounts[np.max(imgr)],
            'darkestShade': np.min(imgr),
            'numOfDarkest': pixelCounts[np.min(imgr)],
            'numOfMedian': numMedian,
            'numAboveMedian': len(lightPixels),
            'numBelowMedian': len(darkPixels),
            'aboveMedianAvg': np.mean(lightPixels),
            'aboveMedianVar': np.var(lightPixels),
            'belowMedianAvg': np.mean(darkPixels),
            'belowMedianVar': np.var(darkPixels),
            'xbar': np.mean(xpos),
            'x2var': np.var(np.mean(xpos, axis=1)),
            'ybar': np.mean(ypos),
            'y2var': np.var(np.mean(ypos, axis=0))
        }, ignore_index = True)
    print('Finished creating ' + whichSet + ' set')
    return df_transform

train = transform_data('train')
test = transform_data('test')
train.to_pickle('data/train.pkl')
test.to_pickle('data/test.pkl')

################################ Exploratory analysis #####################################

print('Creating histograms...')
countNormal = 0
countPneu = 0

avgCountsNormal = [0] * 256
avgCountsPneu = [0] * 256
for index, row in train.iterrows():
    counts, bins = np.histogram(train.iloc[index]['img'].ravel(), 256, (0,255))
    patientType = row['label1']
    if patientType == "Normal":
        avgCountsNormal = avgCountsNormal + counts
        countNormal += 1
    elif patientType == "Pnemonia":
        avgCountsPneu = avgCountsPneu + counts
        countPneu += 1

avgCountsNormal = avgCountsNormal / countNormal
avgCountsPneu = avgCountsPneu / countPneu

plt.bar(np.arange(0,256), avgCountsNormal)
plt.xlabel('Pixel Shade')
plt.ylabel('Avg Count')
plt.savefig('images/hist_normal')

plt.clf()

plt.bar(np.arange(0,256), avgCountsPneu)
plt.xlabel('Pixel Shade')
plt.ylabel('Avg Count')
plt.savefig('images/hist_pneumonia')

plt.clf()

plt.bar(np.arange(0,256), avgCountsNormal, alpha=0.5, label='Normal')
plt.bar(np.arange(0,256), avgCountsPneu, alpha=0.5, label='Pneumonia')
plt.legend(loc='upper right')
plt.savefig('images/hist_overlayed')
print('Finished creating histograms')

plt.clf()

print('Creating boxplots...')
# for every explanatory variable, create boxplot
for col in train.columns:
    if col not in ('img', 'label1', 'label2', 'label3'):
        boxplot = train.boxplot(by = 'label1', column = [col], grid = False)
        plt.title(col)
        plt.suptitle("")
        plt.savefig('images/boxplot_' + col)
print('Finished creating boxplots')