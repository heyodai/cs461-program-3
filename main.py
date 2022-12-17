"""
CS 461 - Program 3

You are given a file with clinical data from the Cleveland Clinic, showing various symptoms and test results for heart patients, with a target variable indicating whether they were eventually diagnosed with heart disease by specific criteria. Your task is to build a neural network to predict the diagnosis based on clinical data.  A full description of the data is given in the attached zip file.

Steps:
1. Load and normalize the data
2. Define the Keras model
3. Train the model
4. Evaluate the model
5. Make predictions
6. Save the model (optional)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich import print

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
0. Define the seed variable

This seed variable allows me to easily experiment with different factors 
and see how they affect the results. I can change the seed value and rerun 
the program to see how the results change.
"""
SEED = {
    'normalize': True,
    '1hot': True, # disabling this causes the model to fail
    'train_test_split': 0.1,
    'random_state': 0,
    'epochs': 100,
    'batch_size': 64,
    'verbose': 0,
    'callbacks': True,
    'dropout': False,
    'dropout_rate': 0.2,
    'save_model': False,
}

"""
1. Load and normalize the data
"""

# 1a. Load the data
df = pd.read_csv('data/raw_cleveland_clinic_data.csv')

# 1b. Remove any rows with missing data or ? values
# TODO: should I keep the ? rows and sub in a value?
df = df.replace('?', np.nan)
df = df.dropna()

# 1c. Normalize the data
x = df.iloc[:,:13].values
y = df.iloc[:,13:14].values

if SEED['normalize']:
    sc = StandardScaler()
    x = sc.fit_transform(x)

    if SEED['1hot']:
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y).toarray()

# 1d. Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = SEED['train_test_split'], 
    random_state = SEED['random_state'],
)

"""
2. Define the Keras model
"""
model = Sequential()
model.add(
    Dense(
        16, 
        input_dim=13, 
        activation='relu',
    )
)
model.add(
    Dense(
        12, 
        activation='relu',
    )
)
model.add(
    Dense(
        5, 
        activation='softmax',
    )
)

"""
3. Train the model
"""

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x_train, 
    y_train, 
    epochs = SEED['epochs'],
    batch_size = SEED['batch_size'],
    verbose = SEED['verbose'],
    # callbacks = SEED['callbacks'],
    # validation_data = (x_test, y_test),
)

"""
4. Evaluate the model
"""

scores = model.evaluate(x_test, y_test)
print(f"{model.metrics_names[1]}: {scores[1]*100}")

"""
5. Make predictions
"""

y_pred = model.predict(x_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

a = accuracy_score(pred,test)
print('Accuracy is:', a*100)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

"""
6. Save the model (optional)
"""
if SEED['save_model']:
    model.save('models/heart_disease.h5')