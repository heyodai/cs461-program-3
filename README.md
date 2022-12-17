# CS 461 - Program 3

## Table of Contents

- [CS 461 - Program 3](#cs-461---program-3)
  - [Table of Contents](#table-of-contents)
  - [Program 3 Report](#program-3-report)
    - [Data Preparation](#data-preparation)
    - [Network Configuration](#network-configuration)
    - [Validation Strategy](#validation-strategy)
    - [Results](#results)
  - [Setup](#setup)
    - [A Note on Apple Silicon](#a-note-on-apple-silicon)

## Program 3 Report

This is my report for Program 3. It's a machine learning program that uses a neural network to predict the presence of heart disease in a patient. The program uses the Cleveland Clinic Heart Disease Dataset from Kaggle.

### Data Preparation

To prepare the data for the Keras model, I first converted the categorical variables into one-hot vectors. I then split the data into training, validation, and test sets. I used a 70/15/15 split for the training, validation, and test sets, respectively. I also standardized the data using the StandardScaler from sklearn.preprocessing.

I later removed the validation set and used k-fold cross validation instead. I used 5 folds for the cross validation.

### Network Configuration

I used a 3-layer neural network with 13 input nodes, 10 hidden nodes, and 1 output node. I used the ReLU activation function for the hidden layer and the sigmoid activation function for the output layer. I used the Adam optimizer with a learning rate of 0.001. I used the binary crossentropy loss function.

I did try a few other configurations, but this one seemed to work the best.

### Validation Strategy

I tried a few different validation strategies. I first tried using a validation set, but I found that the validation accuracy was not very consistent. I then tried using k-fold cross validation, and I found that the validation accuracy was much more consistent.

As far as the test/train split, I used a 70/15/15 split for the training, validation, and test sets, respectively. I later removed the validation set and used k-fold cross validation instead. I used 5 folds for the cross validation.

### Results

I ran the program 10 times with the same configuration and got the following results:

| Run | Accuracy | Precision | Recall | F1 Score |
| --- | -------- | --------- | ------ | -------- |
| 1 | 0.8688524590163934 | 0.8571428571428571 | 0.8571428571428571 | 0.8571428571428571 |
| 2 | 0.8852459016393442 | 0.875 | 0.875 | 0.875 |
| 3 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 4 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 5 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 6 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |

## #Comments

I found that the neural network was able to predict the presence of heart disease with a high degree of accuracy. I also found that the neural network was able to predict the absence of heart disease with a high degree of accuracy. However, I found that the neural network was not very good at predicting the presence of heart disease when the patient did not actually have heart disease. I think this is because the dataset is imbalanced. There are more patients in the dataset that do not have heart disease than patients that do have heart disease. I think this is why the neural network was not very good at predicting the absence of heart disease.

### References

[1] https://www.kaggle.com/ronitf/heart-disease-uci

[2] https://keras.io/

[3] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

[4] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

[5] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

[6] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

[7] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

[8] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

[9] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

[10] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

[11] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

[12] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

[13] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

[14] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

[15] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

[16] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

[17] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html

[18] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

[19] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html

[20] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

[21] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html

[22] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

## Setup

1. Clone the repository
2. Create environment with `make env`

### A Note on Apple Silicon

This program was built on an Apple Silicon system (M2 Macbook Air). It is possible that the program will not run on an Intel-based system. If you are using an Intel-based system, you may need to modify the version of TensorFlow used in the program.

The program uses the `tensorflow-metal` package created by Apple: https://developer.apple.com/metal/tensorflow-plugin/

## Build Commands

- `make env` - Create the environment
- `make run` - Run the program
- `make test` - Run the tests
- `make clean` - Clean the environment 
  - (remove the `venv` directory and Python cache files)

## About the Dataset

*Note: This section is a copy of the text file in the provided heart-data directory. It's originally sourced from [Kaggle](https://www.kaggle.com/datasets/aavigan/cleveland-clinic-heart-disease-dataset).*

### Context

Coronary heart disease (CHD) involves the reduction of blood flow to the heart muscle due to build-up of plaque in the arteries of the heart. It is the most common form of cardiovascular disease. Currently, invasive coronary angiography represents the gold standard for establishing the presence, location, and severity of CAD, however this diagnostic method is costly and associated with morbidity and mortality in CAD patients. Therefore, it would be beneficial to develop a non-invasive alternative to replace the current gold standard.

Other less invasive diagnostics methods have been proposed in the scientific literature including exercise electrocardiogram, thallium scintigraphy and fluoroscopy of coronary calcification. However the diagnostic accuracy of these tests only ranges between 35%-75%. Therefore, it would be beneficial to develop a computer aided diagnostic tool that could utilize the combined results of these non-invasive tests in conjunction with other patient attributes to boost the diagnostic power of these non-invasive methods with the aim ultimately replacing the current invasive gold standard.

In this vein (pun intended), the following dataset comprises 303 observations, 13 features and 1 target attribute. The 13 features include the results of the aforementioned non-invasive diagnostic tests along with other relevant patient information. The target variable includes the result of the invasive coronary angiogram which represents the presence or absence of coronary artery disease in the patient with 0 representing absence of CHD and labels 1-4 representing presence of CHD. Most research using this dataset have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value
0).

The data was collected by Robert Detrano, M.D., Ph.D of the Cleveland Clinic Foundation. See here for protocol specifics.

Also, this paper provides a good summary of the dataset context.

### Content

The data set was downloaded from the UCI website.

**Attribute Information**

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

### Acknowledgements

Robert Detrano, M.D., Ph.D: Principle investigator responsible for collecting data
Inspiration

Diagnosis of Coronary Heart Disease by non-invasive means. 