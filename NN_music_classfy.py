import numpy as np 
import pandas as pd
import os
import random
import operator
import scipy.io.wavfile as wav
from python_speech_features import mfcc

from tqdm import tqdm
import pickle
import scipy
import sys
import pandas as pd
import numpy as np
import IPython
import seaborn as sns
import matplotlib.pyplot as plt

import librosa #Python package for music & audio files
import librosa.display
import librosa.display as lplt

from IPython.display import Audio
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFECV,mutual_info_regression
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

final_data = pd.read_csv("features_3_sec.csv")
final_data = final_data.drop(labels='filename',axis=1)
music_name = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
#To convert categorical data into model-understandable numerica data
class_list = final_data.iloc[:, -1]
convertor = LabelEncoder()
print(convertor)
#Fitting the label encoder & return encoded labels
y = convertor.fit_transform(class_list)


#Standard scaler is used to standardize features & look like standard normally distributed data
fit = StandardScaler()
X = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype = float))
#SPLIT THE DATA INTO TRAINING DATA & TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



#The loss is calculated using sparse_categorical_crossentropy function
def trainModel(model, epochs, optimizer):
    batch_size = 128
    
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, 
                     batch_size=batch_size)

#Plotting the curves
def plotValidate(history):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()

model = nn.Sequential(nn.Linear(X_train.shape[1], 512),
                      nn.ReLU(),
                      nn.Linear(512, 10),
                      nn.Softmax())
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)


losses = []
for epoch in range(5):
    pred_y = model(X_train)
    loss = loss_function(pred_y, y_train)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()

import matplotlib.pyplot as plt
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(0.0001))
plt.savefig('loss_vs_epoch_graph.png')
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128)
print("The test loss is :",test_loss)
print("\nThe test Accuracy is :",test_accuracy*100)


def predict(model, X, y):
    X = X[np.newaxis,...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Expected index: {y}, Predicted index: {predicted_index}")

predict(model, X_test[10], y_test[10])