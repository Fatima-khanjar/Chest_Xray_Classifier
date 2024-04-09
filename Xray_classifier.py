import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

##step2:look to the data
nb_covid_images = len(os.listdir('COVID-19_Radiography_Dataset/COVID/images'))
nb_normal_images = len(os.listdir('COVID-19_Radiography_Dataset/Normal/images'))
print("the number of Covid-19 images: " , nb_covid_images)
print("the number of Normal images: " , nb_normal_images)
# normal_img = cv2.imread('COVID-19_Radiography_Dataset/Normal/images/Normal-10007.png')
# plt.imshow(normal_img)
# print(normal_img.shape())


##step3:Load the data
def loadImages(path, urls, target):
    images = []
    labels = []
    for i in range(len(urls)):
        img_path = path + "/" + urls[i]
        img = cv2.imread(img_path)
        img = img / 255.0
        img = cv2.resize(img, (100,100))
        images.append(img)
        labels.append(target)
    images = np.asarray(images)
    return images, labels

covid_path = "COVID-19_Radiography_Dataset/COVID/images"
covid_url = os.listdir("COVID-19_Radiography_Dataset/COVID/images")
Covid_Images , Covid_labels = loadImages(covid_path,covid_url, 1)      # 1 --> Covid 

normal_path = "COVID-19_Radiography_Dataset/Normal/images"
normal_url = os.listdir("COVID-19_Radiography_Dataset/Normal/images")
Normal_Images , Normal_labels = loadImages(normal_path,normal_url, 0)  # 0 --> normal

#step5:concatenate covid and normal data
data = np.r_[Covid_Images, Normal_Images] # to concatenate arrays
labels = np.r_[Covid_labels, Normal_labels]

#step6:split the concatenated data
x_train, x_test, y_train, y_test = train_test_split(data, labels , test_size=0.25)

#step7:build the CNN model
model = Sequential([
    Conv2D(32, (3,3), input_shape=(100,100,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(16, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(16, (3,3) , activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1,activation="sigmoid")

])

#step7:
model.summary()

#step8:model compile
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
#step9:Train the model
model.fit(x_train, y_train, batch_size = 32, epochs = 5, validation_data = (x_test,y_test))

model.save('Xray_classifier.h5')

#step10:plot the accuracy for the train and test
plt.plot(model.history.history['accuracy'], label = 'train accuracy')
plt.plot(model.history.history['val_accuracy'],label = 'test accuracy')
plt.legend()
plt.show()

#step11:plot the loss for the train and test
plt.plot(model.history.history['loss'], label ='train loss')
plt.plot(model.history.history['val_loss'] , label = 'test loss')
plt.legend()
plt.show()









