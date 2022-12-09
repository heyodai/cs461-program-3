"""
You are given a file with clinical data from the Cleveland Clinic, showing various symptoms and test results for heart patients, with a target variable indicating whether they were eventually diagnosed with heart disease by specific criteria. Your task is to build a neural network to predict the diagnosis based on clinical data.  A full description of the data is given in the attached zip file.

Some thoughts on processing the data:

- This is a small data set--13 features, 303 observations. You do not have sufficient data for a huge network. Use no more than 2 or 3 layers, with a maximum of 5-8 neurons per layer. There's simply no point in constructing a network with dozens of layers, each with hundreds of neurons--this will only guarantee the network will memorize the training data.
- The data is all coded as integers. Some are categorical (1 = yes, 0 = no; 1 = typical angina, 2 = atypical angina, 3 = non-angina pain, 4 = asymptomatic). This data should be left alone, obviously. Other data are true numbers (cholesterol levels; maximum heart rate achieved on stress test). You may want to consider normalizing these variables. (Or not; they're over a small enough range, it may not matter. Try it both ways and see if it makes a difference.)
- For output, you may decide to use two outputs and 1-hot coding (higher output taken as classification), or a single output (linear or hyperbolic tangent), with > 0 or < 0 taken as indicating probable classification.
- If you do any sort of recoding prior to training, you may write a quick Python or Java script doing the transformation. You don't need to submit these "utility" scripts as long as you mention them in your report. (But for many things, such as normalizing numeric variables, there are library functions or API options to do this automatically.)
"""

# TODO: can we remove any of these at the end?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


"""
1. Load the data in from processed_cleveland.csv

Structure of the data:

    age: age in years
    sex: sex (1 = male; 0 = female)
    cp: chest pain type
        Value 1: typical angina
        Value 2: atypical angina
        Value 3: non-anginal pain
        Value 4: asymptomatic 
    trestbps: resting blood pressure (in mm Hg on admission to the
    hospital)
    chol: serum cholestoral in mg/dl
    fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    restecg: resting electrocardiographic results
        Value 0: normal
        Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    thalach: maximum heart rate achieved
    exang: exercise induced angina (1 = yes; 0 = no)
    oldpeak: ST depression induced by exercise relative to rest
    slope: the slope of the peak exercise ST segment
        Value 1: upsloping
        Value 2: flat
        Value 3: downsloping
    ca: number of major vessels (0-3) colored by flourosopy (for calcification of vessels)
    thal: results of nuclear stress test (3 = normal; 6 = fixed defect; 7 = reversable defect)
    num: target variable representing diagnosis of heart disease (angiographic disease status) in any major vessel
        Value 0: < 50% diameter narrowing
        Value 1: > 50% diameter narrowing
"""

# open the file
df = pd.read_csv('processed_cleveland.csv')

# check the data
print(df.head())
print(df.info())
print(df.describe())

# check for missing data
print(df.isnull().sum())

# check for duplicates
print(df.duplicated().sum())

# check for outliers
sns.boxplot(x=df['age'])
plt.show()
sns.boxplot(x=df['trestbps'])
plt.show()
sns.boxplot(x=df['chol'])
plt.show()
sns.boxplot(x=df['thalach'])
plt.show()
sns.boxplot(x=df['oldpeak'])
plt.show()

# check for correlation
sns.heatmap(df.corr(), annot=True)
plt.show()

# check for class imbalance
sns.countplot(x='num', data=df)
plt.show()

# check for correlation with target variable
print(df.corr()['num'].sort_values())

# check for multicollinearity
print(df.corr())

# drop the duplicate
df.drop_duplicates(inplace=True)

# drop the outliers
df.drop(df[df['age'] > 70].index, inplace=True)
df.drop(df[df['trestbps'] > 180].index, inplace=True)
df.drop(df[df['chol'] > 400].index, inplace=True)
df.drop(df[df['thalach'] < 60].index, inplace=True)
df.drop(df[df['oldpeak'] > 5].index, inplace=True)

