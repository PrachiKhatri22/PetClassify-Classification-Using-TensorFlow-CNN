# PetClassify-Classification-Using-TensorFlow-CNN

## Overview
PetClassify is a deep learning-based image classification project that leverages a pre-trained ResNet-50 model in TensorFlow/Keras to classify images of cats and dogs. The model was fine-tuned to perform binary classification using a dataset of labeled images, achieving a high accuracy by utilizing advanced data augmentation techniques and pre-trained weights from ImageNet.

## Technologies Used
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas
- split-folders

## Dataset
The dataset consists of images of cats and dogs, and it is organized into two folders:

- Cat: Contains images of cats.
- Dog: Contains images of dogs.

The dataset was split into 80% for training and 20% for validation using the split-folders package. The images were then preprocessed and resized to 224x224 pixels to match the input size expected by the ResNet-50 model.

## Output
<img width="215" alt="image" src="https://github.com/user-attachments/assets/e3efe37f-7109-4977-8343-e64cabeb0363" />


## Key Features
- Pre-trained ResNet-50 Model: The project uses ResNet-50 as the base model for feature extraction, which is pre-trained on the ImageNet dataset.
- Data Augmentation: Advanced data augmentation techniques, such as rotation, brightness adjustment, shifting, and flipping, were used to enhance the training dataset.
- Fine-tuning: The last few layers of ResNet-50 are fine-tuned to adapt the model for binary classification.
- Visualization: Training and testing accuracy and loss curves are plotted for model evaluation.

## Data Preparation
The data was split into training and validation sets using the split-folders package. The training data was augmented with techniques like:
- Rotation (range of 90 degrees)
- Brightness adjustment (range of 0.1 to 0.7)
- Width and height shift
- Flipping (horizontal and vertical)

## Model Architecture
The model is based on the ResNet-50 architecture, which is a deep convolutional neural network. It is pre-trained on the ImageNet dataset for feature extraction. The model is extended with the following layers:
- GlobalAveragePooling2D: Converts the 3D feature maps into a 1D feature vector.
- Fully Connected Layers (Dense): Adds several dense layers with ReLU activation for learning complex patterns.
- BatchNormalization: Normalizes the activations to speed up the training process.
- Output Layer: A dense layer with a sigmoid activation function for binary classification (Cat vs. Dog).

The following steps were taken to prepare the model:
- The ResNet-50 model was imported with include_top=False, meaning the fully connected layers are excluded.
- The last 143 layers of the ResNet-50 model were frozen to prevent them from being retrained.
- Additional layers were added to fine-tune the model for binary classification.

## Output of Using Transfer Learning (ResNet):
Transfer learning significantly sped up the training process by leveraging the pre-trained ResNet-50 model, which was trained on ImageNet. By using the pre-trained weights, the model could reuse learned low- and high-level features (such as edges, textures, and object parts), eliminating the need to start from scratch. In this project, freezing the first 143 layers of ResNet-50 allowed the model to focus on training only the newly added layers for classifying cats and dogs, reducing the number of parameters to update. This approach reduced the required data, computation, and training time, leading to faster convergence and high accuracy.

<img width="608" alt="image" src="https://github.com/user-attachments/assets/c0524209-1972-46a1-8836-ffab2b35ef5d" />


## Training the Model
The model is compiled using Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and momentum of 0.9. The binary cross-entropy loss function is used for binary classification, and accuracy is tracked as a metric.

## Plotting and Visualizing the performance:
Epoch-wise Training Loss & Accuracy
![image](https://github.com/user-attachments/assets/54d6eb89-3025-4bdc-8c3d-3e540c3f45fa)


## Results
The model achieved a training accuracy of 97.10% and a test accuracy of 97.75%, demonstrating strong performance in classifying images from the Cat vs Dog dataset. The slight difference, with test accuracy being higher than training accuracy, can be attributed to factors such as data augmentation, which introduces randomness during training, and potential random variations in the dataset splits. After training, when the model is given an image, it predicts whether the image is of a cat or a dog, successfully classifying it based on the learned features. Despite the small discrepancy between training and test accuracy, the model showcases effective generalization with high accuracy.

## Output (Successfully classified as Dog) 
![image](https://github.com/user-attachments/assets/764c1c90-87b2-4422-be6d-e4da1c311a00)



