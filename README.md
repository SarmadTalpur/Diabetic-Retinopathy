# Detection of Diabetic Retinopathy using smartphone-based photography

## Introduction

According to WHO (World Health Organization), diabetes is the ninth leading cause of death worldwide. The prevalence of diabetes has been rapidly increasing in low and middle-income countries than in high-income countries. Diabetic retinopathy (DR) is a complication of diabetes caused by high blood sugar levels damaging the back of the eye (retina). DR damages the retina and leads to arbitrary growth of blood vessels. It can make the vessels clot or burst as well. DR can cause blindness if left undiagnosed and untreated.

![image](https://github.com/user-attachments/assets/43c15a0b-502c-433c-a253-c9678ba66f8e)

It is one of the leading causes of blindness worldwide, accounting for more than 3.9 million. However, it takes several years for diabetic retinopathy to reach a stage where it could threaten the sight of a diabetic patient. Most patients remain asymptomatic until they reach advanced stages of DR. The available treatments for visual impairment caused by diabetes have been more effective at slowing visual loss than reversing visual impairment. Hence, early detection of DR before the permanent loss of vision is crucial to ensure better patient health.

## Dataset

We collected fundus image data from the Sindh Institute of Ophthalmology and Visual Sciences (SIOVS), Hyderabad. We merged it with open-source Kaggle competition data to detect diabetic retinopathy. Finally, we created a dataset of 35,122 images divided into two classes: DR and No DR. 9,313 images were DR fundus, and 25,809 were non-DR fundus.

## Preprocessing

The images varied considerably since the open-source data was collected from different sources. Some were zoomed in, and some zoomed out. The fundus ratio of an eye to the image had to be set for all images. Hence, we created a Python script to preprocess all images, cut the extra black background of all the images, and set the standard resolution for all images, i.e., 300* 300.

We separated 790 images from the data for testing. Later, we divided the rest of the data into 80% training and 20% validation datasets.

## Algorithms

We trained three different deep learning models with transfer learning.

### 1. VGG16
VGG16 consists of 16 layers with weights, 13 of which are convolution layers and 3 of which are dense layers. Five Max Pooling Layers are used that do not hold weights. The VGG16 model accepts a tensor of size 224*224 with 3 RGB channels. We prepared the dataset using an image data generator and used data augmentation to increase the data size.
We downloaded the VGG16 model and fine-tuned it. We removed the default output dense layer and added a new dense output layer to predict from 2 classes (DR or No DR). We used the ‘sigmoid’ activation function for the output layer.
We compiled the model using the ‘Adam’ optimizer, with a default learning rate of 0.001 and a ‘binary_crossentropy’ loss function. We used an early stopping callback to monitor validation accuracy in the training process. This means the training process will end if validation accuracy doesn’t improve after a certain number of epochs. We also used the restoration of weight property on early stopping to use the weights from the epoch, which performed best on validation data.

### 2. ResNet50
The input tensor used by Resnet50 has a dimension of 224*224. Thus, we created the dataset by employing an image data generator, preprocessing the photos to reduce their size, and then adding data augmentation by rotating, flipping, zooming in, and adjusting the width and height of the images. To address the issue of DR categorization, we applied transfer learning to the downloaded ResNet50 model and fine-tuned it to add a few extra layers. We added the following layers in the respective order:
• Flatten
• Dropout layer
• Dense layer
• Dropout
• Dense output layer
We made our finely adjusted Resnet model to learn on the last block while freezing the default weights of the Resnet50 model trained on the ImageNet dataset.
We built the model using an Adam optimizer with a learning rate 0.0001 and a binary cross-entropy loss function.
To recover the best weights and to stop the training process after 12 epochs, we employed Early Stopping Callback. We used a batch size of 64 for both the training and validation datasets for doing the training.

### 3. Custom CNN Model
A deep learning model with 16 layers consists of seven convolutional layers, three pooling layers, a flattening layer, followed by a pair of dropout and dense layers, and an output dense layer.

![image](https://github.com/user-attachments/assets/3fcad359-928f-4dc6-a676-37a6557c23ee)

The model accepts an input image of size 224 * 224. Conv 1 layers use 64 filter size with kernel size 3. Conv 2 uses 128 filters with a 3 * 3 kernel size. Conv 3 uses a filter size of 256 with kernel size 3. For all pooling layers, max-pooling layers with a pool size and stride size of 2 were used. Dropout layers are used with a factor of 0.3. The last dense layer is used for classification and uses the sigmoid activation function. All the convolutional layers are used with the same padding, i.e., the image size won’t be reduced due to convolution operation.
We prepared the dataset using an image data generator. We used image data augmentation on training and validation data. We compiled the model using the ‘Adam’ optimizer. We used the ‘binary_crossentropy’ loss function. For efficient training, we used early stopping by checking the improvement in validation accuracy and also defined to restore the best weights. We trained and validated the model using training and validation data using 200 epochs and early stopping callback.

## Results

### VGG16
The VGG16 Model was trained for 100 epochs. Approaching the 23rd epoch, the model validation accuracy remained constant and did not improve. Therefore, the training process was stopped due to an early stopping callback. Configurations and weights were restored from the optimal epoch, i.e., epoch 11.

### ResNet50
After 30 epochs, model training stopped with an early stop callback. Validation accuracy didn’t improve for a few epochs, and the callback interrupted the training. The best model weights were restored from epoch 18.

### Custom CNN Model
Model training occurred successively until 14 epochs. After the 14th epoch, the model training was force-stopped due to an early stopping callback when approaching the highest validation accuracy. Additionally, the model restored weights from epoch two due to optimal performance.

| Model | Training Accuracy (%) | Training Loss (%) | Validation Accuracy (%) | Validation Loss (%) | AUC (%) |
|----------------|-----------|-------------|----------------------------| ------------- | ------------- |
| `VGG16` | 73.72 | 57.67 | 74.53 | 55.94 | 61.74 |
| `ResNet50` | 73.99 | 57.62 | 74.08 | 58.72 | 54.64 |
| `Custom CNN Model` | 73.99 | 57.36 | 74.05 | 57.26 | 50.00 |



After comparing the results of all the trained deep learning models, VGG16 outperformed other models by obtaining the highest validation accuracy, i.e., 74.53%, and the lowest validation loss of 55.94%. Thus, VGG16 performed best on the dataset provided and is deployed in the smartphone application.
