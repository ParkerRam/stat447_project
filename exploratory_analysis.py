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
    all_label = ''
    if row['Label'] == 'Normal':
        all_label = 'Healthy'
    elif row['Label_1_Virus_category'] == 'bacteria':
        all_label = 'Bacteria'
    elif row['Label_2_Virus_category'] == 'COVID-19':
        all_label = 'COVID-19'
    else:
        all_label = 'Other Virus'

    df.at[index, 'all_label'] = all_label

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
                                   rotation_range=30,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=False)

test_datagen = ImageDataGenerator()

PIXELS_RESIZE = 200
train_flow = train_datagen.flow_from_dataframe(dataframe=df_train,
                                               directory='data/train',
                                               x_col='X_ray_image_name',
                                               y_col='all_label',
                                               target_size=(PIXELS_RESIZE, PIXELS_RESIZE),
                                               color_mode='grayscale',
                                               class_mode='raw')

test_flow = test_datagen.flow_from_dataframe(dataframe=df_test,
                                             directory='data/test',
                                             x_col='X_ray_image_name',
                                             y_col='all_label',
                                             target_size=(PIXELS_RESIZE, PIXELS_RESIZE),
                                             color_mode='grayscale',
                                             class_mode='raw')

################################ Transform data #####################################
median_pixel = round(255 / 2)

"""
Calculates summary statistics of given image and returns dictionary for all features
Params:
    imgr: array representation of image
    label: the classification of given image
Returns:
    summary_dict: dictionary of features for given image
"""
def img_summary(imgr, label):
    # for calculation of stats
    unique, counts = np.unique(imgr, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    img_pixels = imgr.ravel()

    # light if > median; dark if < median
    light_pixels = img_pixels[img_pixels > median_pixel]
    dark_pixels = img_pixels[img_pixels < median_pixel]

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

    num_median = 0
    if median_pixel in pixel_counts.keys():
        num_median = pixel_counts[median_pixel]

    summary_dict = {
        'img': imgr,
        'label': label,
        'lungStatus': 'Healthy' if label == 'Healthy' else 'Pneumonia',
        'grayToneAvg': np.mean(imgr),
        'grayToneVar': np.var(imgr),
        'lightestGrayTone': np.max(imgr),
        'numOfLightest': pixel_counts[np.max(imgr)],
        'darkestGrayTone': np.min(imgr),
        'numOfDarkest': pixel_counts[np.min(imgr)],
        'numOfMedian': num_median,
        'numAboveMedian': len(light_pixels),
        'numBelowMedian': len(dark_pixels),
        'aboveMedianAvg': np.mean(light_pixels),
        'aboveMedianVar': np.var(light_pixels),
        'belowMedianAvg': np.mean(dark_pixels),
        'belowMedianVar': np.var(dark_pixels),
        'xbar': np.mean(xpos),
        'x2bar': np.var(xpos),
        'ybar': np.mean(ypos),
        'y2bar': np.var(ypos),
        'x2ybr': np.mean(x2ypos),
        'xy2br': np.mean(xy2pos),
        'xybar': xybar
    }
    return summary_dict

"""
Transforms images in given data augmentation object 
Params:
    image_aug_iter: data augmentation iterator to augment images and transform to array representation
    batch: number of batches of images to transform
Returns:
    df_transform: transformed set
"""
def transform_data(image_aug_iter, batch):
    print('Creating set...')
    df_transform = pd.DataFrame()

    count = 0
    for x in image_aug_iter:
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
count_normal = 0
count_pneu = 0

avg_count_normal = [0] * 256
avg_counts_pneu = [0] * 256
for index, row in train.iterrows():
    counts, bins = np.histogram(train.iloc[index]['img'].ravel(), 256, (0,255))
    patient_type = row['lungStatus']
    if patient_type == "Healthy":
        avg_count_normal = avg_count_normal + counts
        count_normal += 1
    elif patient_type == "Pneumonia":
        avg_counts_pneu = avg_counts_pneu + counts
        count_pneu += 1

avg_count_normal = avg_count_normal / count_normal
avg_counts_pneu = avg_counts_pneu / count_pneu

plt.bar(np.arange(0,256), avg_count_normal)
plt.xlabel('Pixel Gray Tone')
plt.ylabel('Avg Count')
plt.savefig('images/hist_normal')

plt.clf()

plt.bar(np.arange(0,256), avg_counts_pneu)
plt.xlabel('Pixel Gray Tone')
plt.ylabel('Avg Count')
plt.savefig('images/hist_pneumonia')

plt.clf()

plt.bar(np.arange(0,256), avg_count_normal, alpha=0.5, label='Normal')
plt.bar(np.arange(0,256), avg_counts_pneu, alpha=0.5, label='Pneumonia')
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