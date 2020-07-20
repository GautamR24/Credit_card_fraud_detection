
# importing the libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# importing the dataset
data = pd.read_csv('creditcard.csv')

# checking whether the data is imbalanced or not
fraud = data[data['Class']==1]
valid = data[data['Class']==0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print(len(data[data['Class']==1]))
print(len(data[data['Class']==0]))

# getting info about the fraud transactions 
print(fraud.Amount.describe())

# getting the info about the valid transactions
print(valid.Amount.describe())

# produsing a correlation matrix which will help us to predict important 
# features relevant for predictions
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8,square=True)

# Dividing the data
# using X we will make the predictions then 
# then the result will be compared with Y for accuracy
X = data.drop(['Class'],axis=1)
Y = data["Class"]

# getting the info about the dimensions
print(X.shape)
print(Y.shape)


# splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size = 0.2,random_state = 42)


# training the random forest model using xTrain and yTrain
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xTrain,yTrain)
yPred = rfc.predict(xTest)

# calculating the accuracy score
from sklearn.metrics import accuracy_score  
acc = accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc)) 
  















