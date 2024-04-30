import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from catboost import CatBoostClassifier
#from xgboost import XGBClassifier

# Load data from Excel file
data = pd.read_excel(r"C:\Users\mones\OneDrive\Documents\project\project\phd_QuestionAnswers_extractive_fasttext_embeddings 1.xlsx")

# Assuming 'Label' is the column you want to encode (target variable)
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Separate features (X) and target (y)
X = data.iloc[:, :-1]
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers with default hyperparameters for multi-class classification
classifiers = {
    'Perceptron': Perceptron(),
    'MLP': MLPClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),
    #'CatBoost': CatBoostClassifier(verbose=0),  # Adjust verbosity if needed
    #'XGBoost': XGBClassifier()
}

# Define hyperparameter grids for Perceptron and MLP (adjust as needed)
perceptron_param_grid = {
    'alpha': np.linspace(0.0001, 0.01, 100),
    'max_iter': np.arange(100, 1000, 100),
    'penalty': [None, 'l1', 'l2', 'elasticnet']
}

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': np.linspace(0.001, 0.01, 10)
}

# Define evaluation metrics for multi-class classification
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
}

results = {}

# Function to perform hyperparameter tuning and evaluate classifiers
def tune_and_evaluate(classifier_name, classifier, param_grid):
    print(f"Tuning hyperparameters for {classifier_name}...")
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    classifier.set_params(**best_params)
    
    print(f"Evaluating {classifier_name}...")
    scores = {}
    for metric_name, metric_func in metrics.items():
        score = cross_val_score(classifier, X_train, y_train, cv=5, scoring=make_scorer(metric_func))
        scores[metric_name] = np.mean(score)
    results[classifier_name] = scores

# Tune and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    if classifier_name == 'Perceptron':
        tune_and_evaluate(classifier_name, classifier, perceptron_param_grid)
    elif classifier_name == 'MLP':
        tune_and_evaluate(classifier_name, classifier, mlp_param_grid)
    else:
        tune_and_evaluate(classifier_name, classifier, {})  # Use empty param_grid for classifiers with default parameters

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print(results_df)