import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

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

# Create multinomial label
for index, row in df.iterrows():
    allLabel = ''
    if row['Label'] == 'Normal':
        allLabel = 'Healthy'
    elif row['Label_1_Virus_category'] == 'bacteria':
        allLabel = 'Bacteria'
    elif row['Label_2_Virus_category'] == 'COVID-19':
        allLabel = 'COVID-19'
    else:
        allLabel = 'Other Virus'

    df.at[index, 'allLabel'] = allLabel

df_train = df[df['Dataset_type'] == 'TRAIN']
df_test = df[df['Dataset_type'] == 'TEST']

df_train.to_pickle('data/train_metadata.pkl')
df_test.to_pickle('data/test_metadata.pkl')

################################ Image Augmentation #####################################
train_datagen = ImageDataGenerator(featurewise_center=False,
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

test_datagen = ImageDataGenerator()

PIXELS_RESIZE = 200
train_flow = train_datagen.flow_from_dataframe(dataframe = df_train,
                                               directory = 'data/train',
                                               x_col = 'X_ray_image_name',
                                               y_col = 'allLabel',
                                               target_size = (PIXELS_RESIZE, PIXELS_RESIZE),
                                               color_mode = 'grayscale',
                                               class_mode = 'raw')

test_flow = test_datagen.flow_from_dataframe(dataframe = df_test,
                                             directory = 'data/test',
                                             x_col = 'X_ray_image_name',
                                             y_col = 'allLabel',
                                             target_size = (PIXELS_RESIZE, PIXELS_RESIZE),
                                             color_mode = 'grayscale',
                                             class_mode = 'raw')

################################ Transform data #####################################
medianPixel = round(255 / 2)
def img_summary(imgr, label):
    # for calculation of stats
    unique, counts = np.unique(imgr, return_counts=True)
    pixelCounts = dict(zip(unique, counts))

    imgPixels = imgr.ravel()

    # light if > median; dark if < median
    lightPixels = imgPixels[imgPixels > medianPixel]
    darkPixels = imgPixels[imgPixels < medianPixel]

    xpos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
    ypos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
    x2ypos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
    xy2pos = np.empty((PIXELS_RESIZE, PIXELS_RESIZE))
    for x in range(PIXELS_RESIZE):
        for y in range(PIXELS_RESIZE):
            xpos[x,y] = x*(imgr[x,y]/255)
            ypos[x,y] = y*(imgr[x,y]/255)
            x2ypos[x,y] = (x*x)*y*(imgr[x,y]/255)
            xy2pos[x,y] = x*(y*y)*(imgr[x,y]/255)

    n = PIXELS_RESIZE*PIXELS_RESIZE
    xybar = ((n * np.sum(xpos*ypos)) - (np.sum(xpos)*np.sum(ypos))) / (np.sqrt((n*np.sum(xpos**2) - np.sum(xpos)**2) * (n*np.sum(ypos**2) - np.sum(ypos)**2)))

    numMedian = 0
    if medianPixel in pixelCounts.keys():
        numMedian = pixelCounts[medianPixel]

    summary_dict = {
        'img': imgr,
        'label': label,
        'lungStatus': 'Healthy' if label == 'Healthy' else 'Pneumonia',
        'grayToneAvg': np.mean(imgr),
        'grayToneVar': np.var(imgr),
        'lightestGrayTone': np.max(imgr),
        'numOfLightest': pixelCounts[np.max(imgr)],
        'darkestGrayTone': np.min(imgr),
        'numOfDarkest': pixelCounts[np.min(imgr)],
        'numOfMedian': numMedian,
        'numAboveMedian': len(lightPixels),
        'numBelowMedian': len(darkPixels),
        'aboveMedianAvg': np.mean(lightPixels),
        'aboveMedianVar': np.var(lightPixels),
        'belowMedianAvg': np.mean(darkPixels),
        'belowMedianVar': np.var(darkPixels),
        'xbar': np.mean(xpos),
        'x2bar': np.var(xpos),
        'ybar': np.mean(ypos),
        'y2bar': np.var(ypos),
        'x2ybr': np.mean(x2ypos),
        'xy2br': np.mean(xy2pos),
        'xybar': xybar
    }
    return summary_dict

def transform_data(imageAugIter, batch):
    print('Creating set...')
    df_transform = pd.DataFrame()

    count = 0
    for x in imageAugIter:
        count = count + 1
        if count > batch:
            break

        images = x[0]
        labels = x[1]

        for img, label in zip(images, labels):
            summary_dict = img_summary(img.reshape(PIXELS_RESIZE, PIXELS_RESIZE), label)

            df_transform = df_transform.append({
                'img': summary_dict['img'],
                'label': summary_dict['label'],
                'lungStatus': summary_dict['lungStatus'],
                'grayToneAvg': summary_dict['grayToneAvg'],
                'grayToneVar': summary_dict['grayToneVar'],
                'lightestGrayTone': summary_dict['lightestGrayTone'],
                'numOfLightest': summary_dict['numOfLightest'],
                'darkestGrayTone': summary_dict['darkestGrayTone'],
                'numOfDarkest': summary_dict['numOfDarkest'],
                'numOfMedian': summary_dict['numOfMedian'],
                'numAboveMedian': summary_dict['numAboveMedian'],
                'numBelowMedian': summary_dict['numBelowMedian'],
                'aboveMedianAvg': summary_dict['aboveMedianAvg'],
                'aboveMedianVar': summary_dict['aboveMedianVar'],
                'belowMedianAvg': summary_dict['belowMedianAvg'],
                'belowMedianVar': summary_dict['belowMedianVar'],
                'xbar': summary_dict['xbar'],
                'x2bar': summary_dict['x2bar'],
                'ybar': summary_dict['ybar'],
                'y2bar': summary_dict['y2bar'],
                'x2ybr': summary_dict['x2ybr'],
                'xy2br': summary_dict['xy2br'],
                'xybar': summary_dict['xybar'],
            }, ignore_index = True)
    return df_transform

train = transform_data(train_flow, 300)
test = transform_data(test_flow, 19)
train.to_pickle('data/train.pkl')
test.to_pickle('data/test.pkl')

################################ Check Class Balances #####################################
print('Training data summary:')
print(train['label'].value_counts())
print('Testing data summary:')
print(test['label'].value_counts())

################################ Example images #####################################

# Create images directory
try:
    os.mkdir('images/')
except OSError:
    print ('Creation of the directory images/ failed or already created')
else:
    print ('Successfully created the directory images/')

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

################################ Exploratory analysis #####################################

print('Creating histograms...')
countNormal = 0
countPneu = 0

avgCountsNormal = [0] * 256
avgCountsPneu = [0] * 256
for index, row in train.iterrows():
    counts, bins = np.histogram(train.iloc[index]['img'].ravel(), 256, (0,255))
    patientType = row['lungStatus']
    if patientType == "Healthy":
        avgCountsNormal = avgCountsNormal + counts
        countNormal += 1
    elif patientType == "Pneumonia":
        avgCountsPneu = avgCountsPneu + counts
        countPneu += 1

avgCountsNormal = avgCountsNormal / countNormal
avgCountsPneu = avgCountsPneu / countPneu

plt.bar(np.arange(0,256), avgCountsNormal)
plt.xlabel('Pixel Gray Tone')
plt.ylabel('Avg Count')
plt.savefig('images/hist_normal')

plt.clf()

plt.bar(np.arange(0,256), avgCountsPneu)
plt.xlabel('Pixel Gray Tone')
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
for col in df_train.columns:
    if col not in ('img', 'label', 'lungStatus'):
        boxplot = train.boxplot(by = 'label', column = [col], grid = False)
        plt.title(col)
        plt.suptitle("")
        plt.savefig('images/boxplot_' + col)
print('Finished creating boxplots')