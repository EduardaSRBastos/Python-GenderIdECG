# -*- coding: utf-8 -*-
"""
@author: Eduarda
"""
#imports and global variables
import numpy as np
import os
import csv
import pandas as pd
import random
from scipy import stats
from scipy.signal import find_peaks
import json
samples = 219

data_path = 'C:/Users/Duda Bastos/Dropbox/Laboratório/Capturas Verão com Ciência 2021/Capturas'

#read all files for random sample and print info
a = random.randint(1,samples)
sample_path = os.path.join(data_path,str(a))
json_path = os.path.join(sample_path,str(a)+'.json')
#read data and visualize
with open(json_path, 'r') as f:
    sample_data =  json.load(f)
    user_id = sample_data['ID']
    user_gender = sample_data['Gender']
    user_age = sample_data['Age']
    user_weight = sample_data['Weight']
    user_height = sample_data['Height']
    user_education = sample_data['Education']
    user_occupation = sample_data['Occupation']
    user_region = sample_data['Region']
    user_covid = sample_data['COVID-19_vaccinated']
    user_gymfreq= sample_data['Gym_frequency']
    user_sport = sample_data['Sport_practice']
    user_freqexerc = sample_data['Frequency_of_exercise']
    user_exercsdur = sample_data['Exercise_session_duration']
    user_prefexrc = sample_data['Prefered_exercises']
    user_specdiet = sample_data['Specific_diet']
    user_fruitsveg = sample_data['Fruits-vegetables_per_day']
    user_junkfood = sample_data['Candy-Salty_snacks-soda_consumption']
    user_wcoms = sample_data['Water_consumption']
    user_hprobl = sample_data['Health_problems']
    user_dmed = sample_data['Daily_medication']
    user_shab = sample_data['Smoking_habits']
    user_shours = sample_data['Sleeping_hours']
    user_abc = sample_data['Alcoholic_beverage_consumption']
    sensors = sample_data['sensors']['name']
    print("Sample User Characteristics:")
    print("User Id:",user_id)
    print("User Gender:",user_gender)
    print("User Age:",user_age)
    print("User Occupation:",user_occupation)
    df = pd.read_csv(os.path.join(sample_path,str(a)+".txt"),'\t', skiprows=3)
    if(len(df.columns)>1):
        df = df.iloc[:,5:6]
    graf = df.plot(title='ECG', figsize=(30,5))
    graf.legend(['ECG'])
    df.hist()

from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn
import requests

genderfile_csv = "https://raw.githubusercontent.com/impires/JupyterNotebooksECGData/main/Dataset-features/TabelaGender.csv"
df_gender = pd.read_csv(genderfile_csv)
df_gender.columns = df_gender.columns.str.replace("Average of ", "")
df_gender.drop('ID', inplace=True, axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn

train, test = train_test_split(df_gender, test_size=0.3)

X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

#Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
]
alllabels = ['Male', 'Female']
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print('\n........ Score and Classification Report for {0} .............\n'.format(name))
    scores.append(score)
    print(classification_report(y_test, y_pred))

    cfm1 = confusion_matrix(y_test, y_pred)
    seaborn.heatmap(cfm1, xticklabels=alllabels, yticklabels=alllabels,cmap = 'YlGnBu')
    plt.title('Confusion matrix: {0}' .format(name), fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
print("|{0:^25}|{1:^25}|".format("Classifier","Score"))
print("------------------------------------------------------")
for name,score in zip(names,scores):
    print("|{0:^25}|{1:^25}|".format(name,score))

