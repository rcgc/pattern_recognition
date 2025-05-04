import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# 2. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Feature scaling (important for SVMs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Hyperparameter tuning using GridSearchCV (for RBF kernel)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

# 5. Train the SVM model with RBF kernel using GridSearchCV for tuning
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print("Best parameters found:", grid_search.best_params_)

# 6. Use the best model
best_model = grid_search.best_estimator_

# 7. Predict and evaluate
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 8. Plot decision boundaries (for 2D visualization)
def plot_decision_boundaries(X, y, model, class_names):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict using the model for the mesh grid (this uses only the 2D data for visualization)
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(yy.ravel())]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.title('SVM Decision Boundary (RBF Kernel)')

    legend_elements = scatter.legend_elements()[0]
    plt.legend(legend_elements, class_names, title="Classes")
    plt.grid(True)

# Combine training and test data for plotting (using only first two features for visualization)
X_combined = np.vstack((X_train[:, :2], X_test[:, :2]))  # Use only the first 2 features
y_combined = np.hstack((y_train, y_test))

# 9. Plot confusion matrix using matplotlib
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # Annotate the confusion matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.show()

# Call plotting functions
plot_decision_boundaries(X_combined, y_combined, best_model, class_names)

# Display confusion matrix using matplotlib
plot_confusion_matrix(cm, class_names)
