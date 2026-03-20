# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# a) Read dataset
data = pd.read_csv("IRIS.csv")

print(data.head())

# b) Scatter plot
plt.scatter(data['sepal_length'], data['sepal_width'], c=data['species'].astype('category').cat.codes)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Width")
plt.show()

# c) Split data
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# d) Fit model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# e) Predict new data
new_data = [[5, 3, 1, 0.3]]
prediction = model.predict(new_data)

print("Predicted Species:", prediction)
