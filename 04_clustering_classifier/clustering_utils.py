import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def plot_feature_space(dataset: pd.DataFrame, x_col:str = 'temporal_frequency', 
                       y_col:str = 'orientation', figsize: tuple = (12, 4), 
                       title:str = 'Feature Space'):
    """
    Plots a scatterplot of the feature space.

    Parameters:
    - dataset: dataset containing the data.
    - x_col: name of the feature for the x-axis.
    - y_col: name of the feature for the y-axis.
    - figsize: tuple, size of the figure.
    - title: str, title of the plot.

    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=dataset[x_col], y=dataset[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
def apply_kmeans_clustering(dataset: pd.DataFrame, n_clusters: int, x_col: str = 'temporal_frequency', 
                            y_col: str = 'orientation', random_state: int = 42) -> pd.DataFrame:
    """
    Applies K-Means clustering to the dataset and plots the result.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the data to be clustered. It must contain the features used for clustering.
    - n_clusters: The number of clusters to form. This parameter defines how many groups the K-Means algorithm will try to create.
    - x_col: The name of the feature to use for the x-axis in the plot (default is 'temporal_frequency').
    - y_col: The name of the feature to use for the y-axis in the plot (default is 'orientation').
    - random_state: A seed for the random number generator, ensuring reproducibility of results (default is 42).

    Returns:
    - pd.DataFrame: The dataset with an additional column 'KMeans_Cluster' indicating the cluster assignment for each data point.
    """
    X = dataset[[x_col, y_col]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    dataset['KMeans_Cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(12, 4))
    sns.scatterplot(x=dataset[x_col], y=dataset[y_col], hue=dataset['KMeans_Cluster'],
                    palette='viridis', legend=False)
    plt.title('K-Means Clustering')
    plt.show()

    return dataset
def apply_hierarchical_clustering(dataset: pd.DataFrame, n_clusters: int, x_col: str = 'temporal_frequency', 
                                  y_col: str = 'orientation') -> pd.DataFrame:
    """
    Applies Agglomerative (Hierarchical) Clustering to the dataset and plots the result.

    Parameters:
    - dataset: The dataset containing the data to be clustered. It must contain the features used for clustering.
    - n_clusters: The number of clusters to form. This parameter defines how many groups the hierarchical clustering algorithm will try to create.
    - x_col: The name of the feature to use for the x-axis in the plot (default is 'temporal_frequency').
    - y_col: The name of the feature to use for the y-axis in the plot (default is 'orientation').

    Returns:
    - The dataset with an additional column 'Hierarchical_Cluster' indicating the cluster assignment for each data point.
    """
    X = dataset[[x_col, y_col]]
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    dataset['Hierarchical_Cluster'] = agglo.fit_predict(X)

    plt.figure(figsize=(12, 4))
    sns.scatterplot(x=dataset[x_col], y=dataset[y_col], hue=dataset['Hierarchical_Cluster'],
                    palette='coolwarm', legend=False)
    plt.title('Hierarchical Clustering')
    plt.show()

    return dataset
def plot_area_cluster_heatmaps(dataset: pd.DataFrame, area_col: str = 'area', kmeans_col: str = 'KMeans_Cluster', 
                               hier_col: str = 'Hierarchical_Cluster') -> None:
    """
    Plots heatmaps showing the relationship between brain areas and clustering results.

    Parameters:
    - dataset: pandas DataFrame containing clustering and area columns.
    - area_col: Name of the column representing brain areas.
    - kmeans_col: Name of the column with KMeans clustering labels.
    - hier_col: Name of the column with Hierarchical clustering labels.
    """
    plt.figure(figsize=(14, 6))

    # KMeans Heatmap
    plt.subplot(1, 2, 1)
    kmeans_ct = pd.crosstab(dataset[area_col], dataset[kmeans_col])
    sns.heatmap(kmeans_ct, annot=True, fmt='d', cmap='Blues')
    plt.title('KMeans Cluster vs Area')
    plt.xlabel('KMeans Cluster')
    plt.ylabel('Brain Area')

    # Hierarchical Heatmap
    plt.subplot(1, 2, 2)
    hier_ct = pd.crosstab(dataset[area_col], dataset[hier_col])
    sns.heatmap(hier_ct, annot=True, fmt='d', cmap='Greens')
    plt.title('Hierarchical Cluster vs Area')
    plt.xlabel('Hierarchical Cluster')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

def train_area_models(dataset: pd.DataFrame, features: list, label_col: str = 'active', area_col: str = 'area', 
                      test_size: float = 0.2, random_state: int = 10)  -> tuple:
    """
    Trains XGBoost models per area and returns models, results, and a fitted LabelEncoder.

    Parameters:
    - dataset: pandas DataFrame containing the data.
    - features: List of feature column names to be used in training the model.
    - label_col: Name of the column containing the target variable (default is 'active').
    - area_col: Name of the column containing the area labels (default is 'area').
    - test_size: Proportion of the dataset to include in the test split (default is 0.2).
    - random_state: Random seed for reproducibility (default is 10).

    Returns:
    - models (dict): A dictionary mapping each area to its corresponding trained XGBoost model.
    - results (dict): A dictionary mapping each area to a dictionary containing 'X_test', 'Y_test', and 'Y_pred'.
    - label_encoder: A fitted LabelEncoder used to encode the target variable.
    """
    models = {}
    results = {}
    label_encoder = LabelEncoder()

    for area, group in dataset.groupby(area_col):
        print(f"Training model for area: {area}")
        
        X = group[features]
        Y = group[label_col]

        # Skip if only one class is present
        if Y.nunique() < 2:
            print(f"Skipping area {area} — only one class present: {Y.unique()}")
            continue

        # Check if each class has at least two samples for stratification
        if Y.value_counts().min() < 2:
            print(f"Skipping area {area} — insufficient samples for stratified split (class counts: {Y.value_counts().to_dict()})")
            continue

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=Y
        )
        label_encoder.fit(Y_train)

        model = xgb.XGBClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        models[area] = model
        results[area] = {
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_pred': Y_pred
        }

    return models, results, label_encoder

def evaluate_and_plot_results(results: dict, label_encoder: LabelEncoder, n_cols: int = 3, figsize_per_plot: int = 6)  -> None:
    """
    Evaluates classification results per area and visualizes confusion matrices.

    Parameters:
    - results: dict of evaluation results per area. Each entry must contain 'X_test', 'Y_test', and 'Y_pred'.
    - label_encoder: A fitted LabelEncoder instance used for axis tick labels in confusion matrices.
    - n_cols: Number of columns in the subplot grid (default is 3).
    - figsize_per_plot: Scale factor for each subplot in inches (default is 6).
    """
    warnings.filterwarnings('ignore')  # suppress sklearn warnings like undefined metrics

    areas = list(results.keys())
    num_areas = len(areas)

    n_rows = math.ceil(num_areas / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot, n_rows * figsize_per_plot))
    axs = axs.flatten() if num_areas > 1 else [axs]

    for idx, area in enumerate(areas):
        res = results[area]
        Y_test = res['Y_test']
        Y_pred = res['Y_pred']

        accuracy = accuracy_score(Y_test, Y_pred)
        class_report = classification_report(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        print(f"Classification Report for Area: {area}\n{class_report}\n")

        ax = axs[idx]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        title_text = f"Area: {area}\nAccuracy: {accuracy:.4f}"
        if len(label_encoder.classes_) == 2:
            try:
                auc_score = roc_auc_score(Y_test, Y_pred)
                title_text += f" | ROC-AUC: {auc_score:.4f}"
            except Exception:
                pass
        ax.set_title(title_text)

    # Remove unused subplots
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
def compare_with_baseline(results: dict) -> dict:
    """
    Compares model accuracy with a baseline (most frequent class) per area.

    Parameters:
    - results: dict containing 'X_test', 'Y_test', and 'Y_pred' for each area.

    Returns:
    - summary: dict with model and baseline accuracy for each area.
    """
    summary = {}

    for area, res in results.items():
        X_test = res['X_test']
        Y_test = res['Y_test']
        Y_pred = res['Y_pred']

        model_accuracy = accuracy_score(Y_test, Y_pred)

        dummy_clf = DummyClassifier(strategy='most_frequent')
        dummy_clf.fit(X_test, Y_test)
        Y_dummy_pred = dummy_clf.predict(X_test)
        dummy_accuracy = accuracy_score(Y_test, Y_dummy_pred)

        print(f"Area: {area}")
        print(f"Model Accuracy: {model_accuracy:.4f}")
        print(f"Baseline (Most Frequent Class) Accuracy: {dummy_accuracy:.4f}")

        if model_accuracy <= dummy_accuracy + 0.01:
            print("⚠️ Warning: Model is not performing better than the baseline.\n")
        else:
            print("✅ Model is performing better than the baseline.\n")

    return summary

def determine_active(group: pd.DataFrame) -> int:
    """
    Determines if a group of units is active based on the percentage of units 
    that have a spike rate above a certain threshold.
    
    Parameters:
    - group: DataFrame containing spike data for a specific area and stimulus.

    Returns:
    - 1 if the percentage of active units is >= 30%, else 0.
    """
    total_units = group['unit_id'].nunique()  
    active_units = group[group['spikes_per_second'] > 10]['unit_id'].nunique()  
    
    if total_units == 0:
        return 0
    
    return 1 if (active_units / total_units) * 100 >= 30 else 0