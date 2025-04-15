import math
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def plot_feature_space(dataset: pd.DataFrame, x_col: str = 'temporal_frequency', 
                       y_col: str = 'orientation', figsize: tuple = (12, 4), 
                       title: str = 'Feature Space'):
    """
    Plots a scatterplot of the feature space with a log scale on the x-axis.

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
    plt.xscale('log')  
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

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dataset[x_col], y=dataset[y_col], hue=dataset['KMeans_Cluster'],
                    palette='viridis', legend=False)
    plt.xscale('log')  
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

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dataset[x_col], y=dataset[y_col], hue=dataset['Hierarchical_Cluster'],
                    palette='coolwarm', legend=False)
    plt.xscale('log')  
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

def train_area_models_regressor(dataset: pd.DataFrame, features: list, label_col: str = 'proportion_active_units', area_col: str = 'area', 
                                test_size: float = 0.2, random_state: int = 10) -> tuple:
    """
    Trains a regression model for each area in the dataset.
    Parameters:
    - dataset: DataFrame containing the data.
    - features: List of feature columns to use for training.
    - label_col: Column name for the target variable.
    - area_col: Column name for the area.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random seed for reproducibility.
    Returns:
    - models: Dictionary of trained models for each area.
    - results: Dictionary containing test data and predictions for each area.
    """
    models = {}
    results = {}

    for area, group in dataset.groupby(area_col):
        print(f"Training model for area: {area}")
        
        X = group[features]
        Y = group[label_col]

        if Y.nunique() < 2:
            print(f"Area {area} skipped due to all equal values: {Y.unique()}")
            continue

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )

        model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        models[area] = model
        results[area] = {
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_pred': Y_pred
        }

    return models, results

def evaluate_and_plot_results(results: dict, n_cols: int = 3, figsize_per_plot: int = 6) -> None:
    """
    Evaluates regression metrics and plots the results for each area.
    Parameters:
    - results: dict containing 'X_test', 'Y_test', and 'Y_pred' for each area.
    - n_cols: Number of columns for the plot grid.
    - figsize_per_plot: Size of each subplot.
    Returns:
    - None (displays the plots).
    """
    warnings.filterwarnings('ignore')  
    areas = list(results.keys())
    num_areas = len(areas)
    n_rows = math.ceil(num_areas / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot, n_rows * figsize_per_plot))
    if num_areas == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for idx, area in enumerate(areas):
        res = results[area]
        Y_test = res['Y_test']
        Y_pred = res['Y_pred']

        mse = mean_squared_error(Y_test, Y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        print(f"Regression Metrics: {area}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  R^2 Score: {r2:.4f}\n")

        ax = axs[idx]
        ax.scatter(Y_test, Y_pred, alpha=0.7, label="Prediction")

        min_val = min(min(Y_test), min(Y_pred))
        max_val = max(max(Y_test), max(Y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], '--', color='red', label="Perfect Prediction")

        ax.set_title(f"Area: {area}\nMSE: {mse:.4f} | RMSE: {rmse:.4f}\nMAE: {mae:.4f} | R²: {r2:.4f}")
        ax.set_xlabel("Real Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()

    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def compare_with_baseline(results: dict) -> dict:
    """
    Compares model performance with a baseline regressor that predicts the mean of the target variable per area.

    Parameters:
    - results: dict containing 'X_test', 'Y_test', and 'Y_pred' for each area.

    Returns:
    - summary: dict with model and baseline Mean Absolute Error (MAE) for each area.
    """
    summary = {}

    for area, res in results.items():
        X_test = res['X_test']
        Y_test = res['Y_test']
        Y_pred = res['Y_pred']

        model_mae = mean_absolute_error(Y_test, Y_pred)

        dummy_reg = DummyRegressor(strategy='mean')
        dummy_reg.fit(X_test, Y_test)
        Y_dummy_pred = dummy_reg.predict(X_test)
        baseline_mae = mean_absolute_error(Y_test, Y_dummy_pred)

        print(f"Area: {area}")
        print(f"Model MAE: {model_mae:.4f}")
        print(f"Baseline (Mean Predictor) MAE: {baseline_mae:.4f}")

        if model_mae >= baseline_mae: 
            print(f"✅ Model is performing better than the baseline. Delta: {model_mae-baseline_mae}\n")
        else:
            print("⚠️ Warning: Model is not performing better than the baseline.\n")

    return summary
