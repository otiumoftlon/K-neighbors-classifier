import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn import datasets


def ecludian_distance(x1,x2):
    """
    Calculate the Euclidean distance between two points.

    The Euclidean distance between two points in n-dimensional space is 
    the straight-line distance between them, which can be calculated as 
    the square root of the sum of squared differences between their 
    corresponding coordinates.

    Parameters:
    ----------
    x1 : array-like
        Coordinates of the first point. Should be a 1D array or list.
    x2 : array-like
        Coordinates of the second point. Should be a 1D array or list.

    Returns:
    -------
    float
        The Euclidean distance between point x1 and point x2.

    Example:
    -------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([4, 5, 6])
    >>> euclidean_distance(x1, x2)
    5.196152422706632
    """
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1,x2):
    """
    Calculate the Manhattan distance between two points.

    The Manhattan distance (also known as the L1 distance or taxicab 
    distance) between two points in n-dimensional space is the sum of 
    the absolute differences of their corresponding coordinates. 
    It measures how many units one must travel along grid-like paths 
    to get from one point to the other.

    Parameters:
    ----------
    x1 : array-like
        Coordinates of the first point. Should be a 1D array or list.
    x2 : array-like
        Coordinates of the second point. Should be a 1D array or list.

    Returns:
    -------
    float
        The Manhattan distance between point x1 and point x2.

    Example:
    -------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([4, 5, 6])
    >>> manhattan_distance(x1, x2)
    9
    """
    return np.sum(np.abs(x1-x2))


def knn_1(X_train,y_train,point,k=3,algorithm ='ecd'):
    """
    Perform k-Nearest Neighbors (k-NN) classification for a given point.

    This function implements a simple version of the k-NN algorithm to classify 
    a new data point based on the labels of its k-nearest neighbors from a training set. 
    The distance between points can be calculated using either the Euclidean distance ('ecd') 
    or the Manhattan distance ('mad').

    Parameters:
    ----------
    X_train : array-like or DataFrame
        Feature matrix of the training data. Each row represents a data point and 
        each column represents a feature. If a DataFrame is provided, it is converted to a NumPy array.
    y_train : array-like or Series
        Target vector containing the labels for the training data. If a Series is provided, it 
        is converted to a NumPy array.
    point : array-like
        A data point (1D array) to classify.
    k : int, optional, default=3
        The number of nearest neighbors to consider when making the classification.
    algorithm : str, optional, default='ecd'
        The distance metric to use. 'ecd' for Euclidean distance and 'mad' for Manhattan distance.

    Returns:
    -------
    float
        The most common label among the k-nearest neighbors, which is the predicted class for the given point.

    Example:
    -------
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_train = np.array([0, 1, 0])
    >>> point = np.array([4, 4])
    >>> knn_1(X_train, y_train, point, k=2, algorithm='ecd')
    1.0

    Notes:
    ------
    - The function first converts the feature matrix (X_train) and target vector (y_train) to NumPy arrays 
      if they are provided as DataFrame or Series.
    - It computes the distances between the input point and all points in the training set, 
      then selects the k-nearest points based on the chosen distance metric.
    - Finally, it predicts the label by returning the most common label among the k-nearest neighbors.
    """
    m,_ = np.shape(X_train)
    # Convert to NumPy array if necessary
    X_train = X_train.to_numpy()  # Convert feature matrix to NumPy array if it's a DataFrame
    y_train = y_train.to_numpy()  # Convert target vector to NumPy array if it's a Series
    y_train = y_train.reshape(-1, 1)
    # Initialize distances array
    distances = np.zeros((m,1))
    
    # Calculate distances based on the selected algorithm
    if algorithm == 'ecd':
        distances = [ecludian_distance(point,i)  for i in X_train]
    if algorithm == 'mad':
        distances = [manhattan_distance(point,i)  for i in X_train]

    # Find the indices of the k-nearest neighbors
    k_indices = np.argsort(distances)[:k]
    # Extract the labels of the k-nearest neighbors
    k_nearest_labels = [float(y_train[i]) for i in k_indices]
    # Find the most common label
    most_common = Counter(k_nearest_labels).most_common(1)

    return most_common[0][0]    

def knn(X_train,y_train,x_test,k=3,algorithm ='ecd'):
    """
    Perform k-Nearest Neighbors (k-NN) classification for a test set.

    This function applies the k-NN algorithm to classify multiple data points 
    (test set) based on their k-nearest neighbors from a training set. It supports 
    the use of either Euclidean distance ('ecd') or Manhattan distance ('mad') for 
    distance calculations.

    Parameters:
    ----------
    X_train : array-like or DataFrame
        Feature matrix of the training data. Each row represents a data point, and 
        each column represents a feature. If a DataFrame is provided, it is converted 
        to a NumPy array.
    y_train : array-like or Series
        Target vector containing the labels for the training data. If a Series is 
        provided, it is converted to a NumPy array.
    x_test : array-like or DataFrame
        Feature matrix of the test data that needs to be classified. Each row 
        represents a test data point.
    k : int, optional, default=3
        The number of nearest neighbors to consider when making the classification.
    algorithm : str, optional, default='ecd'
        The distance metric to use. 'ecd' for Euclidean distance and 'mad' for Manhattan distance.

    Returns:
    -------
    pred : ndarray
        A 1D NumPy array containing the predicted labels for each data point in the 
        test set.

    Example:
    -------
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_train = np.array([0, 1, 0])
    >>> x_test = np.array([[4, 4], [6, 5]])
    >>> knn(X_train, y_train, x_test, k=2, algorithm='ecd')
    array([1.0, 0.0])

    Notes:
    ------
    - The function first converts the test set (x_test) to a NumPy array if it is provided 
      as a DataFrame.
    - For each test point, the function calls the `knn_1` helper function to predict the 
      label based on its k-nearest neighbors from the training set.
    - The final result is a NumPy array of predictions for the entire test set.
    """
    m,_ = np.shape(x_test)
    x_test = x_test.to_numpy()
    pred = np.zeros((m,1))
    for i in range(m):
        pred[i] = knn_1(X_train,y_train,x_test[i,:],k,algorithm)

    return pred

#Data set of wine
wines = datasets.load_wine()

wine = pd.DataFrame(wines.data,columns=wines.feature_names)

df_stand = (wine - wine.mean())/wine.std()
wine['target'] = wines.target
df_stand['target'] = wines.target

corr_matrix = df_stand.corr()

custom_palette = sns.color_palette(["#257180", "#6c757d", "#FD8B51"]) 

g = sns.pairplot(df_stand, hue='target',
                    x_vars=df_stand.columns[3:6],
                    y_vars=df_stand.columns[9:-1], diag_kind="kde", corner=True,
                    markers=["o", "s", "D"],
                    palette=custom_palette,
                    plot_kws={'alpha': 0.6})

# Set the size of the pairplot figure
g.figure.set_size_inches(8, 6)  # Adjusted size for the pairplot
plt.show()

target = df_stand.pop('target')

#Split the datset
X_train, X_test, y_train, y_test = train_test_split(df_stand,target,test_size=0.3,random_state=42)

#Obtain accuracy score for different ks and the two methods
acc_graf = np.zeros((15,3))
for i in range(15):

    sk_knn = KNeighborsClassifier(n_neighbors=i+1)
    sk_knn.fit(X_train,y_train)
    sk_pred = sk_knn.predict(X_test)
    sk_acc = accuracy_score(y_test,sk_pred)
    predictions = knn(X_train=X_train,y_train=y_train,x_test=X_test,k=i+1,algorithm='ecd')
    my_acc = accuracy_score(y_test,predictions)
    acc_graf[i,:]=[i+1,sk_acc,my_acc]


plt.figure(figsize=(8, 6))

plt.plot(acc_graf[:,0], acc_graf[:,1],marker='o', label='Sklearn', linestyle='-.', color="#257180")

plt.plot(acc_graf[:,0], acc_graf[:,2], marker='s', label='My method', linestyle='--', color="#FD8B51")

plt.title('Accuracy Comparison for Different K Values')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
      