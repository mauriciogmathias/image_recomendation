import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

#set TensorFlow threading options for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#define paths for dataset and features
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/image_recomendation')
dataset_dir = os.path.join(base_dir, 'dataset')
fashion_data_dir = os.path.join(dataset_dir, 'fashion_data')
categories_dir = os.path.join(fashion_data_dir, 'categories')
features_dir = os.path.join(base_dir, 'features')
img_features_path = os.path.join(features_dir, 'image_features.npy')

#load and preprocess images, no labels needed
def load_and_preprocess_dataset(directory, image_size=(224, 224), batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None  #no labels
    )
    #apply resnet preprocessing
    dataset = dataset.map(lambda x: preprocess_input(x))
    return dataset

#load resnet model without top layers, avg pooling for features
def load_resnet50_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

#save features as numpy array
def save_features(features, filename):
    np.save(filename, features)

#load dataset
dataset = load_and_preprocess_dataset(categories_dir)

#load resnet model
resnet_model = load_resnet50_model()
resnet_model.summary()  #check model details

#extract features from images
print("Extracting features from images...")
image_features = resnet_model.predict(dataset, verbose=1)

#save extracted features
save_features(image_features, img_features_path)

print(f"Features saved to {img_features_path}")