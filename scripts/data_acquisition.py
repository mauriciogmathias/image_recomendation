import pandas as pd
from shutil import move
import os
from tqdm import tqdm

#define directories and paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/image_recomendation')
dataset_dir = os.path.join(base_dir, 'dataset')
csv_path = os.path.join(dataset_dir, 'styles.csv')
fashion_data_dir = os.path.join(dataset_dir, 'fashion_data')
images_dir = os.path.join(dataset_dir, 'images')
categories_dir = os.path.join(fashion_data_dir, 'categories')

count = 0

#create the 'fashion_data' directory if it doesn't already exist
if not os.path.exists(fashion_data_dir):
    os.mkdir(fashion_data_dir)

os.chdir(fashion_data_dir)

#load the csv file and process it
data_frame = pd.read_csv(csv_path, usecols=['id','masterCategory']).reset_index()
data_frame['id'] = data_frame['id'].astype('str')

all_images = os.listdir(images_dir)

#create the 'categories' directory if it doesn't exist
if not os.path.exists(categories_dir):
    os.mkdir(categories_dir)

#loop over all images
for image in tqdm(all_images):
    category = data_frame[data_frame['id'] == image.split('.')[0]]['masterCategory']
    category = str(list(category)[0])
    
    #create category directory if it doesn't exist
    if not os.path.exists(os.path.join(categories_dir, category)):
        os.mkdir(os.path.join(categories_dir, category))

    #move the image to the appropriate category directory
    path_from = os.path.join(images_dir, image)
    path_to = os.path.join(categories_dir, category, image)
    move(path_from, path_to)
    
    count += 1  #increment the counter for each image moved

print('Moved {} images.'.format(count))
