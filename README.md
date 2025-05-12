# Iris-Insight-with-KNN

## Project Overview 

**Project Title : Iris insight with KNN**
The goal of this project is to focus on implementing and demonstrating the K-Nearest Neighbors (KNN) algorithm using the famous Iris dataset.

## Objectives
**Primary Goal**:
Classify Iris flowers into three species—Setosa, Versicolor, and Virginica—based on their sepal and petal dimensions.

## Project Structure

### 1. Importing Libraries and Loading the iris dataset
pandas for data manipulation
from sklearn.datasets load the iris dataset directly into our python environment
```python
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
iris
```

### 2. Data processing
**Step-1**
```python
iris.target_names
iris.feature_names
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df
```
This code demonstrates how to extract information about the iris dataset, and organize the data into a pandas DataFrame for analysis.

**Step-2**
```python
df["target"]=iris.target
df
df[df.target==1]
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df
df0=df[:50]
df1=df[50:100]
df2=df[100:150]
df0
```

**Step-3**
```python
import matplotlib.pyplot as plt
%matplotlib inline
```
importing matplotlib for data visualization.
%matplotlib inline is a jupyter notebook command which is used to display plots directly in the notebook output cells.

### 3. Ploting the graph
Create a scatter plot to visualize the two datasets (df0 and df1) with different colors and markers.
```python
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='+')
```

### 4. Train/Test Split
```python
from sklearn.model_selection import train_test_split
x=df.drop(['target','flower_name'],axis='columns')
x
y=df.target
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
len(x_test)
```
test_size=0.2 means 20% of the data will be used for testing, while 80% will be used for training.
random_state=1 ensures the split is reproducible, so every time you run the code, the same data points go into the training and testing sets.

### 5.Model Training
```python
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
```
n_neighbors=10 specifies the number of neighbors the algorithm will use to classify a given data point. In this case, the 10 nearest neighbors are considered.

### 6. Model Prediction
Predictions on the iris dataset.
```python
knn.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
y_predicted=knn.predict(x_test)
y_predicted
cm=confusion_matrix(y_test,y_predicted)
cm
```
The method knn.score(x_test, y_test) is used to evaluate the performance of the K-Nearest Neighbors (KNN) classifier on the test data. 
confusion_matrix(y_test, y_predicted) compares the true labels (y_test) with the predicted labels (y_predicted) and generates a matrix.
the rows represent the true classes, and columns represent the predicted classes:
a. The diagonal shows correct predictions (TP and TN).
b. Off-diagonal values are misclassifications.

## Conclusion
Using the K-Nearest Neighbors (KNN) algorithm, the model was trained and evaluated on the dataset.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]





