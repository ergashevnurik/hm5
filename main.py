# Task 1
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert(node.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None or node.key == key:
            return node
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            temp = self._min_value_node(node.right)
            node.key = temp.key
            node.right = self._delete(node.right, temp.key)
        return node

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def in_order_traversal(self):
        return self._in_order_traversal(self.root, [])

    def _in_order_traversal(self, node, traversal):
        if node is not None:
            self._in_order_traversal(node.left, traversal)
            traversal.append(node.key)
            self._in_order_traversal(node.right, traversal)
        return traversal

# NetworkX visualization function
def plot_tree(bst):
    def build_graph(node, graph, pos=None, x=0, y=0, layer=1):
        if node is not None:
            graph.add_node(node.key)
            pos[node.key] = (x, y)
            if node.left is not None:
                graph.add_edge(node.key, node.left.key)
                build_graph(node.left, graph, pos, x - 1 / layer, y - 1, layer + 1)
            if node.right is not None:
                graph.add_edge(node.key, node.right.key)
                build_graph(node.right, graph, pos, x + 1 / layer, y - 1, layer + 1)

    graph = nx.DiGraph()
    pos = {}
    build_graph(bst.root, graph, pos)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Binary Search Tree Visualization")
    plt.show()

# Construct trees with the given lists
lists = {
    'a': [49, 38, 65, 97, 60, 76, 13, 27, 5, 1],
    'b': [149, 38, 65, 197, 60, 176, 13, 217, 5, 11],
    'c': [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]
}

bst_trees = {}
for name, elements in lists.items():
    bst = BinarySearchTree()
    for elem in elements:
        bst.insert(elem)
    bst_trees[name] = bst
    print(f"Tree '{name}' In-Order Traversal:", bst.in_order_traversal())
    plot_tree(bst)



# Task 2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DecisionTreeClassifierManual:
    def __init__(self, max_depth=None, algorithm="CART"):
        """
        Custom Decision Tree Classifier supporting ID3 and CART algorithms.
        :param max_depth: Maximum depth of the tree.
        :param algorithm: Splitting criterion ('ID3' for Information Gain or 'CART' for Gini Index).
        """
        self.max_depth = max_depth
        self.algorithm = algorithm
        self.tree = None

    def fit(self, X, y):
        """
        Fits the decision tree to the training data.
        :param X: Feature matrix (numpy array).
        :param y: Target vector (numpy array).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predicts the class for each sample in X.
        :param X: Feature matrix (numpy array).
        :return: Predicted classes (numpy array).
        """
        return np.array([self._traverse_tree(sample, self.tree) for sample in X])

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.
        :param X: Feature matrix (numpy array).
        :param y: Target vector (numpy array).
        :param depth: Current depth of the tree.
        :return: A dictionary representing the decision tree.
        """
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1 or depth == self.max_depth or num_samples == 0:
            return {"label": np.bincount(y).argmax()}

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return {"label": np.bincount(y).argmax()}

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if self.algorithm == "ID3":
                    gain = self._information_gain(y, left_indices, right_indices)
                else:  # CART
                    gain = self._gini_gain(y, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, left_indices, right_indices):
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        num_left = sum(left_indices)
        num_right = sum(right_indices)
        total = len(y)

        weighted_avg_entropy = (num_left / total) * left_entropy + (num_right / total) * right_entropy
        return parent_entropy - weighted_avg_entropy

    def _gini_gain(self, y, left_indices, right_indices):
        parent_gini = self._gini(y)
        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        num_left = sum(left_indices)
        num_right = sum(right_indices)
        total = len(y)

        weighted_avg_gini = (num_left / total) * left_gini + (num_right / total) * right_gini
        return parent_gini - weighted_avg_gini

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _traverse_tree(self, sample, node):
        if "label" in node:
            return node["label"]

        if sample[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(sample, node["left"])
        else:
            return self._traverse_tree(sample, node["right"])


# Load SDN Traffic Dataset
file_path = 'SDN_traffic.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Preprocess the dataset
irrelevant_columns = ['id_flow', 'nw_src', 'tp_src', 'nw_dst', 'tp_dst']
data = data.drop(columns=irrelevant_columns, errors='ignore')

non_numeric_columns = data.select_dtypes(include=['object']).columns
for col in non_numeric_columns:
    data[col] = data[col].astype('category').cat.codes

X = data.drop(columns=['category'], errors='ignore').values
y = data['category'].values

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the manual Decision Tree Classifier using ID3
tree_id3 = DecisionTreeClassifierManual(max_depth=3, algorithm="ID3")
tree_id3.fit(X_train, y_train)
predictions_id3 = tree_id3.predict(X_test)

# Train the manual Decision Tree Classifier using CART
tree_cart = DecisionTreeClassifierManual(max_depth=3, algorithm="CART")
tree_cart.fit(X_train, y_train)
predictions_cart = tree_cart.predict(X_test)

# Evaluate both models
from sklearn.metrics import classification_report, confusion_matrix

print("ID3 Classification Report")
print(classification_report(y_test, predictions_id3))

print("CART Classification Report")
print(classification_report(y_test, predictions_cart))


# Confusion Matrix Visualization
def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# Plot confusion matrices
labels = np.unique(y_test)
cm_id3 = confusion_matrix(y_test, predictions_id3, labels=labels)
cm_cart = confusion_matrix(y_test, predictions_cart, labels=labels)

plot_confusion_matrix(cm_id3, "Confusion Matrix - ID3", labels)
plot_confusion_matrix(cm_cart, "Confusion Matrix - CART", labels)


# Task 3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set file path
path = "labeled_flows_xml"  # Replace with the actual path
files = os.listdir(path)

# Initialize data storage
X_Normal = []
Y_Normal = []
X_Attack = []
Y_Attack = []

# Load and preprocess data
for file in files:
    try:
        # Adjust delimiter if necessary
        df = pd.read_csv(os.path.join(path, file), delimiter=',', on_bad_lines='skip')

        # Check for necessary columns
        if 'Tag' in df.columns and 'totalSourceBytes' in df.columns and 'totalDestinationBytes' in df.columns:
            AttackDataframe = df[df['Tag'] == 'Attack']
            NormalDataframe = df[df['Tag'] == 'Normal']

            # Extract features and labels
            X_Normal.append(NormalDataframe[['totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets']])
            Y_Normal.append(NormalDataframe['Tag'])
            X_Attack.append(AttackDataframe[['totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets']])
            Y_Attack.append(AttackDataframe['Tag'])
        else:
            print(f"Skipping file {file} due to missing columns.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Combine data
X_Normal = pd.concat(X_Normal, ignore_index=True)
Y_Normal = pd.concat(Y_Normal, ignore_index=True)
X_Attack = pd.concat(X_Attack, ignore_index=True)
Y_Attack = pd.concat(Y_Attack, ignore_index=True)

# Split data into train and test sets
X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(X_Normal, Y_Normal, test_size=0.3, random_state=42)
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(X_Attack, Y_Attack, test_size=0.3, random_state=42)

# Combine normal and attack samples
X_train = pd.concat([X_train_N, X_train_A], ignore_index=True)
X_test = pd.concat([X_test_N, X_test_A], ignore_index=True)
Y_train = pd.concat([Y_train_N, Y_train_A], ignore_index=True)
Y_test = pd.concat([Y_test_N, Y_test_A], ignore_index=True)

# Train a decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)

# Evaluate the decision tree
Y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(Y_test, Y_pred_dt))

# Confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(Y_test, Y_pred_dt)
sns.heatmap(conf_matrix_dt, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train a random forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, Y_train)

# Evaluate the random forest
Y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(Y_test, Y_pred_rf))

# Confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Task 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/path/to/network_traffic_dataset.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

# Explore the dataset (ensure target and feature selection are appropriate)
print(data.head())

# Feature selection and preprocessing
# Replace 'feature_columns' and 'target_column' with actual column names from the dataset
feature_columns = ['totalSourceBytes', 'totalDestinationBytes', 'totalPackets']  # Example features
target_column = 'weeklyTrafficVolume'  # Example target

X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree Regressor
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regression:")
print(f"Mean Squared Error: {mse_dt}")
print(f"R² Score: {r2_dt}")

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R² Score: {r2_rf}")

# Visualize predictions vs. actual values
def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

plot_predictions(y_test, y_pred_dt, "Decision Tree Predictions vs. Actual")
plot_predictions(y_test, y_pred_rf, "Random Forest Predictions vs. Actual")
