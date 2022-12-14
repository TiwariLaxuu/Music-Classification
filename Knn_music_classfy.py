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

from sklearn.neighbors import KNeighborsClassifier


final_data = pd.read_csv("features_3_sec.csv")
final_data = final_data.drop(labels='filename',axis=1)
music_name = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
#To convert categorical data into model-understandable numerica data
class_list = final_data.iloc[:, -1]
convertor = LabelEncoder()

#Fitting the label encoder & return encoded labels
y = convertor.fit_transform(class_list)


#Standard scaler is used to standardize features & look like standard normally distributed data
fit = StandardScaler()
X = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype = float))
#SPLIT THE DATA INTO TRAINING DATA & TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Original: {music_name[y_test[0]]}   Prediction:: {music_name[y_pred[0]]}')
print(f'Original: {music_name[y_test[2]]}   Prediction:: {music_name[y_pred[2]]}')
print(f'Original: {music_name[y_test[5]]}   Prediction:: {music_name[y_pred[5]]}')
print(f'Original: {music_name[y_test[8]]}   Prediction:: {music_name[y_pred[8]]}')

#Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (16, 9))
sns.heatmap(conf_mat,cmap="BuPu", annot=True, xticklabels = music_name, yticklabels = music_name )