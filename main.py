# Created By: Mayssem Chouaibi
# Description: churn prediction for banking customers
# Dataset available on Kaggle

import dataHelper
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

data = dataHelper.load('churn.csv')

data2 = dataHelper.prepareData(data)

x = data2.drop('Exited', axis='columns')
y = data2['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layers.Dense(len(x_train.columns) - 5, 
                        input_shape=(len(x_train.columns),), 
                        activation='relu'),
     keras.layers.Dense(len(x_train.columns) - 2, 
                        activation='relu'),
    keras.layers.Dense(1, 
                        activation='sigmoid')
])

model.compile(optimizer='Nadam',
                loss='binary_crossentropy', #because output is binary
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)


