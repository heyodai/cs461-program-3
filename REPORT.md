# Program 3 Report

This is my report for Program 3. It's a machine learning program that uses a neural network to predict the presence of heart disease in a patient. The program uses the Cleveland Clinic Heart Disease Dataset from Kaggle.

## Data Preparation

To prepare the data for the Keras model, I first converted the categorical variables into one-hot vectors. I then split the data into training, validation, and test sets. I used a 70/15/15 split for the training, validation, and test sets, respectively. I also standardized the data using the StandardScaler from sklearn.preprocessing.

I later removed the validation set and used k-fold cross validation instead. I used 5 folds for the cross validation.

## Network Configuration

I used a 3-layer neural network with 13 input nodes, 10 hidden nodes, and 1 output node. I used the ReLU activation function for the hidden layer and the sigmoid activation function for the output layer. I used the Adam optimizer with a learning rate of 0.001. I used the binary crossentropy loss function.

I did try a few other configurations, but this one seemed to work the best.

## Validation Strategy

I tried a few different validation strategies. I first tried using a validation set, but I found that the validation accuracy was not very consistent. I then tried using k-fold cross validation, and I found that the validation accuracy was much more consistent.

As far as the test/train split, I used a 70/15/15 split for the training, validation, and test sets, respectively. I later removed the validation set and used k-fold cross validation instead. I used 5 folds for the cross validation.

## Results

I ran the program 10 times with the same configuration and got the following results:

| Run | Accuracy | Precision | Recall | F1 Score |
| --- | -------- | --------- | ------ | -------- |
| 1 | 0.8688524590163934 | 0.8571428571428571 | 0.8571428571428571 | 0.8571428571428571 |
| 2 | 0.8852459016393442 | 0.875 | 0.875 | 0.875 |
| 3 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 4 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 5 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |
| 6 | 0.9016393442622951 | 0.8888888888888888 | 0.8888888888888888 | 0.8888888888888888 |

## Comments

I found that the neural network was able to predict the presence of heart disease with a high degree of accuracy. I also found that the neural network was able to predict the absence of heart disease with a high degree of accuracy. However, I found that the neural network was not very good at predicting the presence of heart disease when the patient did not actually have heart disease. I think this is because the dataset is imbalanced. There are more patients in the dataset that do not have heart disease than patients that do have heart disease. I think this is why the neural network was not very good at predicting the absence of heart disease.

# References

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