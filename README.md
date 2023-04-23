# Machine learning lectures

## Each folder represent my personal growth in the field of Machine Learning and Data Science

### Lecture 1 - Introduction
### Lecture 2 - Regression and Polynomial Regression
### Lecture 3 - KNN and Decision Trees = in KNN we tune K and the number of neighbors
### Lecture 4 - Random Forest, ADAboost and XGBoost, Ensembling, Tune
### Elearning - Data support vectors

## Topics: 

## Decision Trees used to classify the data in a graph 

<img src="https://user-images.githubusercontent.com/114572512/224649987-81984382-c5fc-49dd-86b6-a3704424784a.png" height="200" />



```text
    * Purity (0-1): the lower the better, degree to which a cluster or group contains only a single class or label,
    * Gini impurity index:  measure of the probability of incorrectly classifying a randomly
      chosen element in a dataset if it were randomly labeled according to the distribution of labels in the dataset.
    * Entropy: Entropy is a measure of the degree of disorder or randomness in a system, often associated with
      the amount of information or uncertainty present.
```
<img src="https://user-images.githubusercontent.com/114572512/224651997-08d8a795-3e82-4039-aea6-c7d0ef25d618.png" width="400" />

```text

Advanatages: 
   Very simple works iwth non linear problems 
   interpretable 
   computationally cheap

Problems:
  Unstable ( a change in the data can lead to completely differtent model)
  mostly baed in heristic with no solid  stattisical background
  prone to overfit
  
  
 If i have to classify smth on a graph  that looks like clustering = decision tree

```

## Random Forest
#### Building multiple decision trees from a data set and than taking the mean or majority, PARALLEL = FAST
```text
OOB:Out of bag sampling = Predict the model based on the params not used in the tree
```
<img height="200" alt="image" src="https://user-images.githubusercontent.com/114572512/227628961-93b8d8f0-21be-47a6-b7fd-32a410cf2f63.png">

```text
Feature importance: Decrease impurity = deleting irrelevant features 
```

## ADAboost (Adaptive Boosting)
#### Models built in SEQUENCE = SLOW, working on pricniple of = output of each model is input  to the next 
<img height ="150" alt="image" src="https://user-images.githubusercontent.com/114572512/227629977-41c15c7c-7f5a-49fa-bc79-ce45480ac195.png">


```text
Boosting:  each learning model is trained on a different subset of the data
```
```text
Idea: weak learners on different subsets of the data and combine their predictions to improve the overall accuracy of the model
Usecase: classification, regression, and ranking problems
Benefit: Not usually overfitting
```


## XGBoost (Xtreme Gradient Boosting) 
#### Tree based model that runs in SEQUENCE = SLOW, uses RESIDUAL ERROR from the previous model as input to the next
 <img width="653" alt="image" src="https://user-images.githubusercontent.com/114572512/227632054-1995e5df-617d-4bda-b69f-2d7d281cd294.png">

```text
Advantages: includes regularization techniques, such as L1 and L2 regularization, to prevent overfitting
```

## Parameter Tuning (hyperparameter optimization)   
```text 
* Decision trees: max_depth, min_samples_leaf, min_samples_split, max_features
* Random forest: n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features
* AdaBoost: n_estimators, learning_rate
* XGBoost: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
* KN: K, n_neighbors
```


#### Process of selecting the best set of hyperparameters (set before training the model) for a machine learning algorithm

### Random forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

# Define the hyperparameter space
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform the grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

# Print the best hyperparameters
print(grid_search.best_params_)
```

### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Create the base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)

adaboost = AdaBoostClassifier(base_estimator=base_estimator)

# Define the hyperparameter space
param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.1, 0.5, 1.0]
}

# Perform the grid search
grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

# Print the best hyperparameters
print(grid_search.best_params_)
```
### XGBoost
```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_classifier = xgb.XGBClassifier()

# Define the hyperparameter space
param_grid = {
    'learning_rate': [0.1, 0.5, 1.0],
    'max_depth': [3, 5, 10],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'n_estimators': [50, 100, 200]
}

# Perform the grid search
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

# Print the best hyperparameters
print(grid_search.best_params_)
```
### Support Vector Machine
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(kernel='rbf')

# Define the hyperparameter space
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Perform the grid search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

# Print the best hyperparameters
print(grid_search.best_params_)
```

## Unsupervised Learning

#### Priniple: Finding patterns in data without any labels or target variables 
#### Usecase: Clustering, Dimensionality reduction, Anomaly detection

## Clustering
### Priniple: Finding natural groupings among objects in a dataset
```text
OBJECTIVE
* Maximize similarity within clusters (intra-cluster): cohesive within clusters
* Minimize similarity between clusters (inter-cluster): distinctive between cluster
```
### Main Question: How many clusters should we have?
```text 
* Elbow method: plot the number of clusters against the within-cluster sum-of-squares (WCSS)
* Silhouette score: measure of how similar an object is to its own cluster compared to other clusters
```

### Types of clustering:
1. Paritional algorithms: (K-means, K-medoids) 
2. Hierarchical algorithms: Hierachicla decomposition of the given set of points (Diana, Agnes)
3. Density-based algorithms: Based on connectivityand density = Regions based on density   (DBSCAN, OPTICS)

### Hard vs Soft clustering
```text
Hard clustering: each data point either belongs to only one cluster 
```
```text 
Soft clustering: each data point has a probability of belonging to multiple clusters
```

## K-means
* Unlabelled data
* Principle of: Assigning centroids and than updating them based on the mean of the data points in the cluster
* Objective: Minimize the sum of squared distances between the data points and their assigned clusters
* Each point assigned to the closest centroid

### Steps:
1. Randomly initialize k centroids (random locations)
2. Assign each data point to the closest centroid   
3. Update the centroids to the mean of the data points that are assigned to them
4. Repeat steps 2 and 3 until the centroids don't change

### Choosing the number of clusters 

#### Elbow method
```text
* Plot the number of clusters against the within-cluster sum-of-squares (WCSS)
```
* We are picking the point where the WCSS starts to flatten out (elbow point)
* The elbow point is the optimal number of clusters

#### Silhouette score
```text
* Measure of how similar an object is to its own cluster compared to other clusters
```
* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering
* Takes extremely long to compute
* The higher the silhouette score, the better the clustering

### Evaluation of clustering
```text
* Elbow method = minimising the within-cluster sum-of-squares (WCSS)
```
```text
* Silhouette score = maximising the silhoute score for each cluster
