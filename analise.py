import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import fbeta_score, ConfusionMatrixDisplay, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score


RANDOM_STATE = 33089

# Creating a custom scoring function using fbeta_score with beta=2
fbeta_scorer = make_scorer(fbeta_score, beta=2)

# Loading the dataset with the correct delimiter
file_path = r"files/input/dataset_ic2023.csv"
data = pd.read_csv(file_path, delimiter=";", usecols=['Colesterol Total', 'Idade', 'Glicemia', 'Desfecho'])

# Separating the features (X) and target variable (y)
X = data[['Colesterol Total', 'Idade', 'Glicemia']]
y = data['Desfecho']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


# --- Exploratory Data Analysis --- #

# Merging the features and target variable for the training data
train_data = X_train.copy()
train_data['Desfecho'] = y_train

# Creating a pairplot with the "bright" color palette
sns.pairplot(train_data, hue='Desfecho', diag_kind='kde', palette='bright')
plt.show()

# Calculating the correlation matrix for the training data
correlation_matrix = train_data.corr()

# Plotting the heatmap for the correlation matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.title("Matriz de Correlação")
plt.show()


# --- Training the models --- #

# Função para testar os modelos
def display_confusion_matrix(y, y_pred, title):
    """
    Display the confusion matrix for a given data & model.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    cm = confusion_matrix(y, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=["Positive", "Negative"])
    cmp.plot(ax=ax)
    plt.title(title)
    plt.show()


def plot_decision_tree(X, model):
    # Plotting the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=list(X.columns), class_names=['Negative', 'Positive'], filled=True)
    plt.title("Árvore de Decisão", fontsize=20)
    plt.show()


def compute_bias_variance(estimator, X, y, cv=5):
    # Calculando as estimativas do erro usando validação cruzada
    scores = cross_val_score(estimator, X, y, scoring=fbeta_scorer, cv=cv)

    # Erro é o inverso da acurácia
    errors = [1 - s for s in scores]

    # Viés (Bias) é aproximadamente a média dos erros
    bias = np.mean(errors)

    # Variância é a variação dos erros ao redor da média
    variance = np.var(errors)

    return bias, variance


def test_model(X, y, model):
    y_pred = model.predict(X)

    # Making predictions on the test data
    bias, variance = compute_bias_variance(model, X, y)

    metrics = {
        "bias": bias,
        "variance": variance,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "fbeta_score": fbeta_score(y, y_pred, beta=2)
    }
    return metrics


# Árvore de decisão simples
def train_model_decision_tree(X, y, random_state=None):
    """
    Função para treinar um modelo DecisionTreeClassifier com busca em grid para otimizar os hiperparâmetros.
    """
    # Defining the grid of hyperparameters to search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 7, 10, 15, 20],
        'min_samples_split': [5, 10, 15, 20, 25],
        'min_samples_leaf': [5, 10, 15, 20, 25]
    }

    # Creating the base model to tune
    base_model = DecisionTreeClassifier(random_state=random_state)

    # Using make_scorer to create a custom scoring function
    fbeta_scorer = make_scorer(fbeta_score, beta=2)

    # Instantiating the grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=0, scoring=fbeta_scorer)

    # Fitting the grid search to the data
    grid_search.fit(X, y)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# Regressão Logística
class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    # Custom Logistic Regression Classifier that transforms the predicted values
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=None):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter,
                                        random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        # Adicionando o atributo 'classes_' ao objeto
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict_proba(X)
        y_pred = [1 if proba[1] > 0.5 else -1 for proba in y_pred_proba]
        return y_pred


def train_model_logistic(X, y, random_state=None):
    param_grid_logistic_compatible = {
        'penalty': ['l2', 'none'],
        'C': [0.0001, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
    }

    base_model = CustomLogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid_logistic_compatible,
                               cv=5, n_jobs=-1, verbose=0, scoring=fbeta_scorer)
    grid_search.fit(X, y)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# SVM para classificação
def train_model_SVM(X_train, y_train, random_state=None):
    """
    Função para treinar o modelo SVM.
    """
    # Defining the parameter grid to search
    param_grid = {
        'C': [0.000001, 0.0001, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.0000000001, 0.00000001, 0.0000001, 0.000001, 0.0001, 0.01, 0.1, 0.25, 'scale'],
    }

    # Defining the SVC model
    svc_model = SVC(random_state=random_state)

    # Defining the grid search with cross-validation
    grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid,
                               cv=5, scoring=fbeta_scorer)

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# Random Forests
def train_model_Random_Forest(X_train, y_train, random_state=None):
    """
    Função para treinar o modelo Random Forest.
    """
    # Defining the parameter grid to search
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 60],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }

    # Defining the Random Forest model
    model = RandomForestClassifier(random_state=random_state)

    # Defining the grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                  cv=5, scoring=fbeta_scorer)

    # Fitting the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# Gaussian Mixtures Model
class GMMClassifier(BaseEstimator, ClassifierMixin):
    # Custom GMM Classifier
    def __init__(self, n_components=2, covariance_type='full', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm_pos = None
        self.gmm_neg = None
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_pos = X[y == 1]
        X_neg = X[y == -1]
        self.gmm_pos = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                       random_state=self.random_state)
        self.gmm_neg = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                       random_state=self.random_state)
        self.gmm_pos.fit(X_pos)
        self.gmm_neg.fit(X_neg)
        return self

    def predict(self, X):
        prob_pos = self.gmm_pos.score_samples(X)
        prob_neg = self.gmm_neg.score_samples(X)
        predictions = [1 if p_pos > p_neg else -1 for p_pos, p_neg in zip(prob_pos, prob_neg)]
        return predictions


def train_model_GMM(X_train, y_train, random_state=None):
    """
    Function to train the Gaussian Mixtures Model classifier.
    """
    # Parameters grid for GMM
    param_grid = {
        'n_components': [1, 2, 3, 4],
        'covariance_type': ['full', 'tied', 'diag', 'spherical']
    }

    # Creating a custom scoring function using fbeta_score with beta=2
    fbeta_scorer = make_scorer(fbeta_score, beta=2)

    # Creating GMM classifier
    gmm_classifier = GMMClassifier(random_state=random_state)

    # Creating GridSearchCV
    grid_search = GridSearchCV(gmm_classifier, param_grid, scoring=fbeta_scorer, cv=5)

    # Fitting GridSearchCV
    grid_search.fit(X_train, y_train)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# Extra tree
def train_model_extra_tree(X_train, y_train, random_state=None):
    """
    Trains an ExtraTreesClassifier using GridSearchCV and returns the best model.
    """
    # Parameter grid for ExtraTreesClassifier
    param_grid = {
        'n_estimators': [10, 50, 100, 150],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create the ExtraTreesClassifier
    etc_model = ExtraTreesClassifier(random_state=random_state)

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=etc_model, param_grid=param_grid,
                               cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Returns best estimator
    return grid_search.best_estimator_, grid_search.best_params_


# Dictionary with model functions and their names
models = {
    'Decision Tree': train_model_decision_tree,
    'Logistic Regression': train_model_logistic,
    'SVM': train_model_SVM,
    'Random Forest': train_model_Random_Forest,
    'Gaussian Mixtures': train_model_GMM,
    'Extra Tree': train_model_extra_tree
}

# Loop over each model to train, evaluate, and append results
ls_results = []
for model_name, train_func in models.items():
    trained_model, best_params = train_func(X_train, y_train, random_state=RANDOM_STATE)
    metrics_train = test_model(X_train, y_train, trained_model)
    metrics_test = test_model(X_test, y_test, trained_model)

    results = {
        'Model': model_name,
        'best_params': str(best_params),
        'Bias': metrics_train['bias'],
        'Variance': metrics_train['variance'],
        'Train Accuracy': metrics_train['accuracy'],
        'Test Accuracy': metrics_test['accuracy'],
        'Train Precision': metrics_train['precision'],
        'Test Precision': metrics_test['precision'],
        'Train Recall': metrics_train['recall'],
        'Test Recall': metrics_test['recall'],
        'Train F2-Score': metrics_train['fbeta_score'],
        'Test F2-Score': metrics_test['fbeta_score']
    }
    ls_results.append(results)

# Concatenate all the results into the final dataframe
results_df = pd.DataFrame(ls_results)
results_df.to_csv(r"files\output\results.csv", index=False)


# --- Plotting the results --- #
selected_metrics_train = ['Train Accuracy', 'Train F2-Score']
selected_metrics_test = ['Test Accuracy', 'Test F2-Score']
selected_labels = ['Acurácia', 'F2-Score']
colors = ['blue', 'green']

plt.figure(figsize=(12, 7))

# Plotando pontos para Acurácia
plt.scatter(results_df['Model'], results_df[selected_metrics_train[0]], marker='o', label=f'Treino - {selected_labels[0]}', color=colors[0], s=100, edgecolor='black')
plt.scatter(results_df['Model'], results_df[selected_metrics_test[0]], marker='o', facecolors='none', edgecolors=colors[0], label=f'Teste - {selected_labels[0]}', s=100)

# Plotando pontos para F2-Score
plt.scatter(results_df['Model'], results_df[selected_metrics_train[1]], marker='^', label=f'Treino - {selected_labels[1]}', color=colors[1], s=100, edgecolor='black')
plt.scatter(results_df['Model'], results_df[selected_metrics_test[1]], marker='^', facecolors='none', edgecolors=colors[1], label=f'Teste - {selected_labels[1]}', s=100)

plt.title('Comparação de Acurácia e F2-Score por Modelo')
plt.ylabel('Valor da Métrica')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# Viés e variância
plt.figure(figsize=(12, 8))

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

# Plotando pontos para viés vs. variância com marcador de bola e cores diferentes para cada modelo
for (model, bias, variance), color in zip(results_df[['Model', 'Bias', 'Variance']].values, colors):
    plt.scatter(bias, variance, s=100, label=model, color=color, marker='o')

# Configurações adicionais do gráfico
plt.title('Viés vs. Variância')
plt.xlabel('Viés')
plt.ylabel('Variância')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
