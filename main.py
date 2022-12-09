"""
TODO: add a description of the project
TODO: add a description of the data
TODO: add a description of the model
TODO: add a description of the training process
TODO: add a description of the results

TODO: add steps

You are given a file with clinical data from the Cleveland Clinic, showing various symptoms and test results for heart patients, with a target variable indicating whether they were eventually diagnosed with heart disease by specific criteria. Your task is to build a neural network to predict the diagnosis based on clinical data.  A full description of the data is given in the attached zip file.

Some thoughts on processing the data:

- This is a small data set--13 features, 303 observations. You do not have sufficient data for a huge network. Use no more than 2 or 3 layers, with a maximum of 5-8 neurons per layer. There's simply no point in constructing a network with dozens of layers, each with hundreds of neurons--this will only guarantee the network will memorize the training data.
- The data is all coded as integers. Some are categorical (1 = yes, 0 = no; 1 = typical angina, 2 = atypical angina, 3 = non-angina pain, 4 = asymptomatic). This data should be left alone, obviously. Other data are true numbers (cholesterol levels; maximum heart rate achieved on stress test). You may want to consider normalizing these variables. (Or not; they're over a small enough range, it may not matter. Try it both ways and see if it makes a difference.)
- For output, you may decide to use two outputs and 1-hot coding (higher output taken as classification), or a single output (linear or hyperbolic tangent), with > 0 or < 0 taken as indicating probable classification.
- If you do any sort of recoding prior to training, you may write a quick Python or Java script doing the transformation. You don't need to submit these "utility" scripts as long as you mention them in your report. (But for many things, such as normalizing numeric variables, there are library functions or API options to do this automatically.)
"""
# TODO: consider splitting this into multiple files

# TODO: can we remove any of these at the end?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


"""
1. Load and normalize the data 

Raw data is stored in: raw_cleveland_clinic_data.csv

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

# define enums to make the data more readable
# TODO: Should I remove these?
class Sex:
    MALE = 1
    FEMALE = 0

class ChestPainType:
    TYPICAL_ANGINA = 1
    ATYPICAL_ANGINA = 2
    NON_ANGINAL_PAIN = 3
    ASYMPTOMATIC = 4

class RestingECG:
    NORMAL = 0
    ABNORMAL = 1
    HYPERTROPHY = 2

class Slope:
    UPSLOPING = 1
    FLAT = 2
    DOWNSLOPING = 3

class Thal:
    NORMAL = 3
    FIXED_DEFECT = 6
    REVERSABLE_DEFECT = 7

class Diagnosis:
    LESS_THAN_50 = 0
    GREATER_THAN_50 = 1

# 1a. Load the data into a pandas dataframe
df = pd.read_csv('data/raw_cleveland_clinic_data.csv')

# 1b. Remove any rows with missing data or ? values
# TODO: should I keep the ? rows and sub in a value?
df = df.replace('?', np.nan)
df = df.dropna()
df = df.to_numpy()

# normalize the data
# TODO: add more documentation of normalization function used
# TODO: see if .transform() is better than .fit_transform()
# TODO: try removing normalization altogether
# TODO: try normalizing only the numeric columns
# TODO: try normalizing only the categorical columns
sc = StandardScaler()
data = sc.fit_transform(df)
# print(data)

# also remove the header row
# df = df.drop(0)

# convert from pandas dataframe to a list of dictionaries of numpy arrays
# data = {
    # 'age': df['age'].to_numpy(),
    # 'sex': df['sex'].to_numpy(),
    # 'cp': df['cp'].to_numpy(),
    # 'trestbps': df['trestbps'].to_numpy(),
    # 'chol': df['chol'].to_numpy(),
    # 'fbs': df['fbs'].to_numpy(),
    # 'restecg': df['restecg'].to_numpy(),
    # 'thalach': df['thalach'].to_numpy(),
    # 'exang': df['exang'].to_numpy(),
    # 'oldpeak': df['oldpeak'].to_numpy(),
    # 'slope': df['slope'].to_numpy(),
    # 'ca': df['ca'].to_numpy(),
    # 'thal': df['thal'].to_numpy(),
    # 'num': df['num'].to_numpy(),
# } 
# print(df.to_numpy())
# for row in df:
#     print(row)
#     print(row['age'])
#     print(row['age'].to_numpy())
#     print(type(row['age'].to_numpy()))
    # print()
col_id = {
    'age': 0,
    'sex': 1,
    'cp': 2,
    'trestbps': 3,
    'chol': 4,
    'fbs': 5,
    'restecg': 6,
    'thalach': 7,
    'exang': 8,
    'oldpeak': 9,
    'slope': 10,
    'ca': 11,
    'thal': 12,
    'num': 13,
}
data = [
    {
        'age': int( row[ col_id['age'] ] ),
        'sex': ohe.fitrow[ col_id['sex'] ],
        'cp': row[ col_id['cp'] ],
        'trestbps': row[ col_id['trestbps'] ],
        'chol': row[ col_id['chol'] ],
        'fbs': row[ col_id['fbs'] ],
        'restecg': row[ col_id['restecg'] ],
        'thalach': row[ col_id['thalach'] ],
        'exang': row[ col_id['exang'] ],
        'oldpeak': row[ col_id['oldpeak'] ],
        'slope': row[ col_id['slope'] ],
        'ca': row[ col_id['ca'] ],
        'thal': row[ col_id['thal'] ],
        'num': row[ col_id['num'] ],
    }
    for row in df
]
# data = df.to_dict('list')
# print(data)
for i in range(0, 5):
    print(data[i])



# One-hot encode the categorical columns
# ohe = OneHotEncoder()
# data = ohe.fit_transform(data).toarray()
print(data)

"""
2. Define the Keras model
"""

"""
3. Define loss and optimizer functions
"""

"""
4. Train the model
"""