import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn import datasets


def calculate_accuracy(y_true, y_pred):

    # Ensure the arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Count the number of correct predictions
    correct_predictions =  np.sum([1 if i == j else 0 for i, j in zip(y_true, y_pred)])

    # Calculate the accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy

def ecludian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1,x2):
    return np.sum(np.abs(x1-x2))


def knn_1(X_train,y_train,point,k=3,algorithm ='ecd'):
    m,n = np.shape(X_train)
    X_train = X_train.to_numpy()  # Convert feature matrix to NumPy array if it's a DataFrame
    y_train = y_train.to_numpy()  # Convert target vector to NumPy array if it's a Series
    y_train = y_train.reshape(-1, 1)
    distances = np.zeros((m,1))
    if algorithm == 'ecd':
        distances = [ecludian_distance(point,i)  for i in X_train]
    if algorithm == 'mad':
        distances = [manhattan_distance(point,i)  for i in X_train]

    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [float(y_train[i]) for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)

    return most_common[0][0]    

def knn(X_train,y_train,x_test,k=3,algorithm ='ecd'):
    m,n = np.shape(x_test)
    x_test = x_test.to_numpy()
    pred = np.zeros((m,1))
    for i in range(m):
        pred[i] = knn_1(X_train,y_train,x_test[i,:],k,algorithm)

    return pred


wines = datasets.load_wine()

wine = pd.DataFrame(wines.data,columns=wines.feature_names)

df_stand = (wine - wine.mean())/wine.std()
wine['target'] = wines.target
df_stand['target'] = wines.target

corr_matrix = df_stand.corr()



# Create a subset DataFrame with the first 5 columns and the target
df_subset = df_stand.iloc[:, :5:].copy()  # First 5 columns
df_subset['target'] = df_stand['target'] 

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

X_train, X_test, y_train, y_test = train_test_split(df_stand,target,test_size=0.3,random_state=42)

acc_graf = np.zeros((15,3))
for i in range(15):

    sk_knn = KNeighborsClassifier(n_neighbors=i+1)
    sk_knn.fit(X_train,y_train)
    sk_pred = sk_knn.predict(X_test)
    sk_acc = accuracy_score(y_test,sk_pred)
    predictions = knn(X_train=X_train,y_train=y_train,x_test=X_test,k=i+1,algorithm='ecd')
    my_acc = accuracy_score(y_test,predictions)
    acc_graf[i,:]=[i+1,sk_acc,my_acc]

print(acc_graf)
plt.figure(figsize=(8, 6))

plt.plot(acc_graf[:,0], acc_graf[:,1],marker='o', label='Sklearn', linestyle='-.', color="#257180")

plt.plot(acc_graf[:,0], acc_graf[:,2], marker='s', label='My method', linestyle='--', color="#FD8B51")

plt.title('Accuracy Comparison for Different K Values')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
      