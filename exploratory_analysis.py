import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

summary = pd.read_csv('data/Chest_xray_Corona_dataset_Summary.csv', index_col=0)
df = pd.read_csv('data/Chest_xray_Corona_Metadata.csv', index_col=0)
MEDIAN_PIXEL = round(255 / 2)
PIXELS_RESIZE = 200

"""
Returns training and holdout set after performing data processing and clean up; splits training and test,
updates labels, replaces null, removes uneeded data 
Params:
    df: data frame containing all chest x-ray image data
Returns:
    df_train: data frame containing training set
    df_test: data frame containing test/holdout set
"""
def cleanData(df):
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

    # save to pkl files for later use
    df_train.to_pickle('data/train_metadata.pkl')
    df_test.to_pickle('data/test_metadata.pkl')
    return df_train, df_test

"""
Returns new training and test data sets after performing image augmentation (rotates, shifts, zooms, flips)
Params:
    df_train: data frame containing training set
    df_test: data frame containing test/holdout set
Returns:
    train_flow: update training set but with image augmentation
    test_flow: update holdout set but with image augmentation
"""
def augmentImage(df_train, df_test):
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
    return train_flow, test_flow

"""
Calculates summary statistics of given image and returns dictionary for all features
Params:
    imgr: array representation of image
    label: the classification of given image
Returns:
    summary_dict: dictionary of features for given image
"""
def imgSummary(imgr, label):
    # for calculation of stats
    unique, counts = np.unique(imgr, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    img_pixels = imgr.ravel()

    # light if > median; dark if < median
    light_pixels = img_pixels[img_pixels > MEDIAN_PIXEL]
    dark_pixels = img_pixels[img_pixels < MEDIAN_PIXEL]

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
    if MEDIAN_PIXEL in pixel_counts.keys():
        num_median = pixel_counts[MEDIAN_PIXEL]

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

# Note: TAKES A LONG TIME TO RUN !!!
"""
Transforms images in given data augmentation object 
Params:
    image_aug_iter: data augmentation iterator to augment images and transform to array representation
    batch: number of batches of images to transform
Returns:
    df_transform: transformed set
"""
def transformData(image_aug_iter, batch):
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
            summary_dict = imgSummary(img.reshape(PIXELS_RESIZE, PIXELS_RESIZE), label)

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

"""
Extracts examples of a chest x-ray for a normal, bacterial pneumonia-affect and viral pneumonia-affected patients
images taken from training set 
Params:
    df_train: dataframe containing training data 
"""
def getExamples(df_train):
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

"""
Creates and save a histogram of the distribution of graytones for healthy and pneumonia patients
Params:
    df_train: data frame containing training set 
"""
def grayToneHistogram(df_train):
    print('Creating histograms...')
    count_normal = 0
    count_pneu = 0

    avg_count_normal = [0] * 256
    avg_counts_pneu = [0] * 256
    for index, row in df_train.iterrows():
        counts, bins = np.histogram(df_train.iloc[index]['img'].ravel(), 256, (0,255))
        patient_type = row['lungStatus']
        if patient_type == "Healthy":
            avg_count_normal = avg_count_normal + counts
            count_normal += 1
        elif patient_type == "Pneumonia":
            avg_counts_pneu = avg_counts_pneu + counts
            count_pneu += 1

    avg_count_normal = avg_count_normal / count_normal
    avg_counts_pneu = avg_counts_pneu / count_pneu

    # creates plot for gray tone distirbution of healthy
    plt.bar(np.arange(0,256), avg_count_normal)
    plt.xlabel('Pixel Gray Tone')
    plt.ylabel('Avg Count')
    plt.savefig('images/hist_normal')

    plt.clf()

    # creates plot for gray tone distribution of pnuemonia
    plt.bar(np.arange(0,256), avg_counts_pneu)
    plt.xlabel('Pixel Gray Tone')
    plt.ylabel('Avg Count')
    plt.savefig('images/hist_pneumonia')

    plt.clf()

    # creates overlapped plot of gray tone distribution of healthy and pneumonia
    plt.bar(np.arange(0,256), avg_count_normal, alpha=0.5, label='Normal')
    plt.bar(np.arange(0,256), avg_counts_pneu, alpha=0.5, label='Pneumonia')
    plt.legend(loc='upper right')
    plt.savefig('images/hist_overlayed')
    print('Finished creating histograms')

    plt.clf()


"""
Creates and saves side-by-side boxplots for each feature  
Params:
    df_train: data frame containing training set. Will create box-plot for the distribution of each column in this set 
"""
def getBoxplots(df_train):
    print('Creating boxplots...')
    # for every explanatory variable, create boxplot
    for col in df_train.columns:
        if col not in ('img', 'label', 'lungStatus'):
            boxplot = df_train.boxplot(by = 'label', column = [col], grid = False)
            plt.title(col)
            plt.suptitle("")
            plt.savefig('images/boxplot_' + col)
            plt.clf()
    print('Finished creating boxplots')

########################################################################################################################

"""
Performs data processing and exploratory analysis on chest x-ray data. This includes:
- data transforms and updates
- image augmentation
- feature engineering
- creating side-by-side boxplots for features
"""
def main():
    # clean data
    df_train, df_test = cleanData(df)

    # get example images
    getExamples(df_train)

    # perform image augmentation
    train_flow, test_flow = augmentImage(df_train, df_test)

    # feature engineering - takes very long to run!!!
    train = transformData(train_flow, 300)
    test = transformData(test_flow, 19)
    # save pkl files for future use
    train.to_pickle('data/train.pkl')
    test.to_pickle('data/test.pkl')

    # check class balances
    print('Training data summary:')
    print(train['label'].value_counts())
    print('Testing data summary:')
    print(test['label'].value_counts())

    # exploratory analysis
    # compare distribution of pixel gray tones for healthy vs pneumonia
    grayToneHistogram(train)
    # get side-by-side boxplots of features
    getBoxplots(train)

if __name__ == "__main__":
    main()