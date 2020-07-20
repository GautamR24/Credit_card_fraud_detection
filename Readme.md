#  Project: Credit Card Fraud Detection
In this project we will try to detect the frauds in the transactions.
## Libraries used:
1. pandas
2. seaborn
3. sklearn
4. matplotlib

## Code:
The `creditcard_fraud_detection.py` contains the code.

## Workflow of code:
1. Usually in credit card fraud detection dataset there is lot of imbalance among the data.
2. To get the info about the imbalance , calculate the frauds and valid transactions.
3. it turns out that the fraud transaction are very less as compared to valid therefore there is imbalance.
4. try to produce a correlation matric which will help in finding the features important for making predictions.
5. divide the dataset into two objects one having the transcations only and other having the verdict of the transactions(this will be used to test the accuracy later).
6. divide the dataset into testing and training dataset using the two objects created.
7. train the random forest model and then test it.
8. find the accuracy of the model by comparing it with second second object created above.

## Methodology:
1. import the dataset.
2. check the dataset for data imbalance.
3. produce a correlation matix.
4. store transactions in one object and verdict of transaction in other.
5. divide the dataset into testing and training dataset.
6. train and test the model.
7. find the accuracy.

## Result: 
Random Forest gives an Accuracy of **95%**.