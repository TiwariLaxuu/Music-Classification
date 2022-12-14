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

music_name = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

#define a function to get distance between feature vectors and find neighbors
def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance,trainingset[x],k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#function to identify the nearest neighbors
def nearestclass(neighbors):
    classVote = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
            
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

# define a function that will evaluate a model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(testSet)  

directory = 'genres_original'
# f = open("my.dat", "wb")
# i = 0
# for folder in os.listdir(directory):
#     print(folder)
#     i += 1
#     if i == 11:
#         break
#     for file in os.listdir(directory+"/"+folder):
#         #print(file)
#         try:
#             (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
#             mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
#             covariance = np.cov(np.matrix.transpose(mfcc_feat))
#             mean_matrix = mfcc_feat.mean(0)
#             feature = (mean_matrix, covariance, i)
#             pickle.dump(feature, f)
#         except Exception as e:
#             print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
# f.close()

dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  
    for x in range(len(dataset)):
        if random.random() <split :      
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])  
trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)

leng = len(testSet)
predictions = []
for x in range (leng):
    predictions.append(nearestclass(getNeighbors(trainingSet ,testSet[x] , 5))) 
accuracy1 = getAccuracy(testSet , predictions)
print(accuracy1)

from collections import defaultdict
results = defaultdict(int)

directory = "genres_original"

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1
(rate,sig)=wav.read("genres_original/hiphop/hiphop.00001.wav")
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)
pred = nearestclass(getNeighbors(dataset, feature, 5))
print(f'Orginal Music: HipHop, Prediction Music: {results[pred]}')

(rate,sig)=wav.read("genres_original/country/country.00000.wav")
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)
pred = nearestclass(getNeighbors(dataset, feature, 5))
print(f'Orginal Music: Country, Prediction Music: {results[pred]}')

(rate,sig)=wav.read("genres_original/rock/rock.00000.wav")
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)
pred = nearestclass(getNeighbors(dataset, feature, 5))
print(f'Orginal Music: Rock, Prediction Music: {results[pred]}')

#Confusion matrix
conf_mat = confusion_matrix(testSet, predictions)
plt.figure(figsize = (16, 9))
sns.heatmap(conf_mat,cmap="BuPu", annot=True, xticklabels = music_name, yticklabels = music_name )