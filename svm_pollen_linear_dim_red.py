import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('pollen.csv')

# Split features and labels (assume last column contains the class)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Reduce dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split PCA-transformed features
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train linear SVM on reduced data
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Visualization of PCA-reduced data with decision boundary and support vectors
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', alpha=0.7, label='Train data')
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            edgecolors='k', facecolors='none', linewidths=1.5, label='Support Vectors')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Pollen Data in 2D (PCA) with Support Vectors")
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
plt.show()
