import pandas as pd 
import numpy as np

#the time coulmn is about days follow up
data= pd.read_csv('heart_failure_clinical_records_dataset.csv')

x= data.iloc[:,0:-1].values
y= data.iloc[:,-1].values

#feture scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x= sc.fit_transform(x)
	
#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


#logistic regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#ploting the classifier
from matplotlib import pyplot as plt
plt.figure(figsize=(20,20))
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='rainbow')
plt.title('heart_failure_clinical_records_dataset')
plt.show()


#predicting the test set results
y_pred= classifier.predict(x_test)

#confusion matrix to evaluate the performance of the model
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

