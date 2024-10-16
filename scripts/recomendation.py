import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import matplotlib.pyplot as plt

#set TensorFlow threading options for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#define paths for dataset and features
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/image_recomendation')
features_dir = os.path.join(base_dir, 'features')
dataset_dir = os.path.join(base_dir, 'dataset')
fashion_data_dir = os.path.join(dataset_dir, 'fashion_data')
categories_dir = os.path.join(fashion_data_dir, 'categories')
img_features_path = os.path.join(features_dir, 'image_features.npy')

#load pretrained resnet model for feature extraction
def load_resnet50_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

#load extracted features
def load_features(filename):
    return np.load(filename)

#get all image paths from categories directory
def get_all_image_paths(categories_dir):
    image_paths = []
    for root, dirs, files in os.walk(categories_dir):
        for file in files:
            if file.endswith('.jpg'):  #adjust extension if needed
                image_paths.append(os.path.join(root, file))
    return image_paths

#find similar images using nearest neighbor search
def find_similar_images(query_image_path, model, features, n_neighbors=5):
    #load and preprocess query image
    img = image.load_img(query_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    #extract features for the query image
    query_features = model.predict(img_array)

    #nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(query_features)

    return distances, indices

#plot query and similar images
def plot_similar_images(query_image_path, similar_indices, image_paths):
    #load query image
    query_img = image.load_img(query_image_path)
    
    #create a subplot for query and similar images
    plt.figure(figsize=(15, 10))
    
    #plot query image
    plt.subplot(2, 3, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    
    #plot similar images
    for i, index in enumerate(similar_indices[0]):
        similar_image_path = image_paths[index]
        similar_img = image.load_img(similar_image_path)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(similar_img)
        plt.title(f"Similar Image {i+1}")
        plt.axis('off')
    
    plt.show()

#load resnet model
resnet_model = load_resnet50_model()

#load pre-extracted features
image_features = load_features(img_features_path)

#get all image paths
all_image_paths = get_all_image_paths(categories_dir)

#set query image path
query_image_path = os.path.join(categories_dir, 'Free Items/11212.jpg')

#find similar images
similar_image_distances, similar_image_indices = find_similar_images(query_image_path, resnet_model, image_features, n_neighbors=5)

#plot query and similar images
plot_similar_images(query_image_path, similar_image_indices, all_image_paths)
