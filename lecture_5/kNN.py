# The same code from R to python

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pandas.read_csv("Lecture_06_BreastCancerCoimbra.csv")

#check data

def check_properties(data):
    #Nans
    print("Nans: ", data.isnull().values.any())
    data = data.dropna()
    
    #delete outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    data = data[~((data < (Q1 - 1.5 * (Q3 - Q1))) |(data > (Q3 + 1.5 * (Q3 - Q1)))).any(axis=1)]
    
    return data


def data_preparation(data):
    #Train test split
    train,test = train_test_split(data, test_size=0.2, random_state=42)
    
    #we will be predicting the parameter class
    train_y = train["Class"]
    test_y = test["Class"]
    
    train_x =   train.drop(["Class"], axis=1)
    test_x = test.drop(["Class"], axis=1)
        
    
    return train_x, train_y, test_x, test_y

X_train, y_train, X_test,y_test = data_preparation(data)


#Rescaling the data
def rescaling(X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


scaled_X_train, scaled_X_test = rescaling(X_train, X_test)
scaled = pandas.DataFrame(scaled_X_train, columns=X_train.columns)




    
def kNN(X_train, y_train, X_test, y_test, k):
    pass
if __name__ == "__main__":
    data = check_properties(data)
    