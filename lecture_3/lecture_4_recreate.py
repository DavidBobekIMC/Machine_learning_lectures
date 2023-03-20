#This file will recreate the lecture 4 example from R to Python

#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#set seed 1
np.random.seed(1)
#Import the data
data = pd.read_csv('lecture_3\Lecture_04_Banknotes.txt')

#Insights into the data

"""
# Banknotes data
# 1. variance of Wavelet Transformed image (continuous)
# 2. skewness of Wavelet Transformed image (continuous)
# 3. curtosis of Wavelet Transformed image (continuous)
# 4. entropy of image (continuous)
# 5. class (0 is genuine, 1 is forged)

"""


#Create a decision tree regressor to predict the class

def check_data(data):
    data = data.dropna()
    
    #OUTLIER DETECTION
    #Check for outliers
    #Boxplot
    #delete outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.3 * IQR)) |(data > (Q3 + 1.3 * IQR))).any(axis=1)]
    #plt.show()
    return data

#data = check_data(data)
data.boxplot()
plt.show()
#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:4], data.iloc[:,4], test_size=0.2, random_state=42)

#Create the decision tree regressor
tree_decision = DecisionTreeRegressor()

#Fit the model
tree_decision.fit(X_train, y_train)
plt.figure(figsize=(20,10))
plt.plot(X_train, y_train, 'o', color='black')


#Predict the values
y_pred = tree_decision.predict(X_test)

#Check the accuracy
accuracy = tree_decision.score(X_test, y_test)
print(accuracy)

#Visualize the tree with matplotlib

from sklearn import tree
fig, ax = plt.subplots(figsize=(20,20))
tree.plot_tree(tree_decision, fontsize=8, ax=ax)
plt.show()

