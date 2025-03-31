import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
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


def initialize_cache(data_dir: str) -> EcephysProjectCache:
    """
    Initialize the AllenSDK EcephysProjectCache object for the given data directory.

    Parameters:
    - data_dir: The path to the data directory containing the manifest file.

    Returns:
    - An EcephysProjectCache object initialized with the manifest file.
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    return EcephysProjectCache.from_warehouse(manifest=manifest_path)

def get_session_data_from_sessionid(session_id: int, cache: EcephysProjectCache):
    """
    Get the EcephysSession object for the given session ID.

    Parameters:
    - session_id: The ID of the session to retrieve.
    - cache: An EcephysProjectCache object.

    Returns:
    - An EcephysSession object containing the session data.
    """
    return cache.get_session_data(session_id)

def stimulus_spike_table(stimuli_table: pd.DataFrame, 
                         columns: list, session, 
                         units: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of spike rates for each stimulus presentation and unit.

    Parameters:
    - stimuli_table: A DataFrame containing stimulus presentation data.
    - columns: A list of columns to include in the output table.
    - session: An EcephysSession object.
    - units: A DataFrame containing unit data.

    Returns:
    - A DataFrame containing the spike rates for each stimulus presentation and unit.
    """
    stimuli_ids = stimuli_table.index
    print(f"Stimuli IDs: {stimuli_ids}")
    spikes_per_stimulus = session.presentationwise_spike_times(stimulus_presentation_ids = stimuli_ids)
    spikes_per_stimulus.reset_index(inplace=True)
    
    spike_counts = spikes_per_stimulus.groupby(['stimulus_presentation_id', 'unit_id'])['spike_time'].count().reset_index()
    spike_counts.rename(columns={'spike_time': 'spike_count'}, inplace=True)
    
    dataset = pd.merge(spike_counts, stimuli_table[columns + ['duration']], on='stimulus_presentation_id')
    dataset['spikes_per_second'] = dataset['spike_count'] / dataset['duration']
    dataset = dataset.drop(columns=['duration', 'spike_count'])
    dataset['area'] = units.loc[dataset['unit_id'], 'ecephys_structure_acronym'].values

    return dataset

def replace_not_recognized_nulls(dataset: pd.DataFrame, drop_nan: bool = False):
    """
    Replace all values that are not recognized with np.nan.
    Prints the number of replacements if any are found.

    Parameters:
    - dataset: The DataFrame to process.
    - drop_nan: If True, drop rows containing NaN values.

    Returns:
    - The processed DataFrame with unrecognized values replaced with np.nan.
    """
    unrecognized_values = ['null', 'NULL', '', ' ']
    
    mask = dataset.isin(unrecognized_values)
    count = mask.sum().sum()  
    
    if count > 0:
        print(f"Replacing {count} unrecognized values with np.nan.")
    
    dataset = dataset.replace(unrecognized_values, np.nan)
    
    if drop_nan:
        dataset = dataset.dropna()
        print("Dropped rows containing NaN values.")

    return dataset

def count_units_by_area(all_areas: set, session_table: pd.DataFrame, 
                        cache: EcephysProjectCache) -> dict:
    """
    Returns a dictionary where the keys are session IDs, and the values are dictionaries of 
    brain areas with their respective unit counts for each session.

    Parameters:
    - all_areas: A set of brain areas to count units for.
    - session_table: A DataFrame containing session data.
    - cache: An EcephysProjectCache object.

    Returns:
    - A dictionary with session IDs as keys and dictionaries of area unit counts as values.
    The inner dictionaries have brain areas as keys and the number of units in that area as values.
    """
    session_area_unit_counts = {}
    units = cache.get_units()
    
    for sid in session_table.index.values:
        session_counts = {}
        
        for area in all_areas:
            maskunits = (units["ecephys_session_id"] == sid) & (units["ecephys_structure_acronym"] == area)
            units_in_area = len(units[maskunits])
            
            session_counts[area] = units_in_area
        
        session_area_unit_counts[sid] = session_counts
    
    return session_area_unit_counts

def get_areas_from_session(session_id: int, cache: EcephysProjectCache) -> list:
    """
    Get a list of all brain areas with units in the given session.

    Parameters:
    - session_id: The ID of the session to check.
    - cache: An EcephysProjectCache object.

    Returns:
    - A list of unique brain areas with units in the specified session.
    """
    units = cache.get_units()
    areas = units[units['ecephys_session_id'] == session_id]['ecephys_structure_acronym'].unique()
    return areas

def get_all_areas(cache: EcephysProjectCache) -> set:
    """
    Get a set (unique elements) of all brain areas with units in the dataset.

    Parameters:
    - cache: An EcephysProjectCache object.
    
    Returns:
    - A set of unique brain areas with units in the dataset.
    """
    all_areas = set()
    for sid in cache.get_session_table().index.values:
        areas = get_areas_from_session(sid, cache)  
        all_areas.update(areas)
    return all_areas

def filter_sessions_by_numerical_column(session_table: pd.DataFrame, 
                                        column: str, min_value: float = 0, 
                                        max_value: float = np.inf) -> pd.DataFrame:
    """
    Filter the session table by a numerical column.

    Parameters:
    - session_table: A DataFrame containing session data.
    - column: The name of the column to filter by.
    - min_value: The minimum value for the filter (inclusive).
    - max_value: The maximum value for the filter (inclusive).

    Returns:
    - A DataFrame containing only the sessions that meet the filter criteria.
    """
    mask = (session_table[column] >= min_value) & (session_table[column] <= max_value)
    print(f"Filtered out {len(session_table) - mask.sum()} sessions.")
    return session_table[mask]

def compute_entropy(row: pd.Series) -> float:
    """
    Compute the entropy  H = -sum(p * log2(p)) of a distribution for a session.

    Parameters:
    - row: A Series containing the counts of each category.

    Returns:
    - The entropy of the distribution.
    """
    counts = row.values

    total = counts.sum()
    
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    
    return -np.sum(probabilities * np.log2(probabilities))

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