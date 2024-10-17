# image_recomendation 

This project implements an image recommendation system using deep learning techniques. It extracts features from images of fashion products and uses these features to recommend similar images based on a query image.

## Dataset

The dataset used for this project is the ["Fashion Product Images Small"](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle, which contains images of various fashion products along with metadata such as category labels.

## Project Structure

The project is divided into three main scripts:

1. **data_acquisition.py**: 
   - Loads the dataset and organizes the images into directories based on their `masterCategory`.
   - Moves images into corresponding category folders for easier processing.

2. **feature_extractor.py**: 
   - Uses the ResNet50 model (pre-trained on ImageNet) to extract image features.
   - The extracted features are saved as a `.npy` file to be used for image similarity search.

3. **recommendation.py**: 
   - Loads the pre-extracted features and allows a user to input a query image.
   - Uses Nearest Neighbor search to find and display the most similar images based on the query image.

## Pretrained Model

The project uses the ResNet50 model for feature extraction. However, an observed limitation in this project is that ResNet50 extracts the white background of images as a prominent feature. As a result, the recommended images may not always be visually similar to the query image but may share a common white background. While this is a known flaw, it provided valuable insights during testing.

## How to Run the Project

1. **Data Acquisition**:
   - Download the dataset from Kaggle and place it in the specified directory structure.
   - Run the `data_acquisition.py` script to organize the images into categories.

2. **Feature Extraction**:
   - Run the `feature_extractor.py` script to extract and save the image features using ResNet50.

3. **Image Recommendation**:
   - Run the `recommendation.py` script, specify a query image, and see the recommended similar images.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- tqdm
