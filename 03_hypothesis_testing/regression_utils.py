import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt

def fit_and_test_models(X, y, unit_id, area, covariate, plot=False):
    """
    Fit both linear and sinusoidal models and test their significance.
    
    Parameters:
    X : array-like, shape (n_samples,)
        The input time points
    y : array-like, shape (n_samples,)
        The target values
    plot : bool, optional
        If True, plot the results. Default is False.
        
    Returns:
    dict: Dictionary containing test results
    """
    # Reshape X for sklearn
    X_linear = X.reshape(-1, 1)
    
    # Fit linear model
    linear_model = LinearRegression()
    linear_model.fit(X_linear, y)
    y_pred_linear = linear_model.predict(X_linear)
    
    # Calculate R² and p-value for linear model
    r2_linear = r2_score(y, y_pred_linear)
    slope = linear_model.coef_[0]
    
    # Calculate p-value for linear model
    n = len(X)
    slope_stderr = np.sqrt(np.sum((y - y_pred_linear) ** 2) / (n-2)) / np.sqrt(np.sum((X - np.mean(X)) ** 2))
    t_stat = slope / slope_stderr
    p_value_linear = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
    
    # Create features for sinusoidal model
    X_sin = np.column_stack([
        np.ones_like(X),
        np.sin(180*X/(2*np.pi)),
        np.cos(180*X/(2*np.pi)),
    ])
    
    # Fit sinusoidal model
    sin_model = LinearRegression(fit_intercept=False)
    sin_model.fit(X_sin, y)
    y_pred_sin = sin_model.predict(X_sin)
    
    # Calculate R² and F-statistic for sinusoidal model
    r2_sin = r2_score(y, y_pred_sin)
    f_stat = (r2_sin / 2) / ((1 - r2_sin) / (n - 3))
    p_value_sin = 1 - stats.f.cdf(f_stat, 2, n-3)
    
    if plot:
        # Plot linear fit
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, alpha=0.5, label='Data')
        plt.plot(X, y_pred_linear, color='red', label=f'Linear fit (R²={r2_linear:.3f})')
        plt.xlabel(f'{covariate}')
        plt.ylabel('spikes_per_second')
        plt.title(f'Linear Regression: unit {unit_id}, area {area}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot sinusoidal fit
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, alpha=0.5, label='Data')
        sort_idx = np.argsort(X)
        plt.plot(X[sort_idx], y_pred_sin[sort_idx], color='red', 
                 label=f'Sinusoidal fit (p-value={p_value_sin:.3f})')
        plt.xlabel(f'{covariate}')
        plt.ylabel('spikes_per_second')
        plt.title(f'Sinusoidal Regression: unit {unit_id}, area {area}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'linear_r2': r2_linear,
        'linear_p_value': p_value_linear,
        'slope': slope,
        'slope_direction': 'positive' if slope > 0 else 'negative',
        'sin_r2': r2_sin,
        'sin_p_value': p_value_sin
    }

def analyse_area(dataset, area, covariate, plot):
    """
    Analyze the relationship between a covariate and neural activity for a specific brain area.

    Parameters:
    dataset : pandas.DataFrame
        The dataset containing neural activity data. Must include columns 'area', 'unit_id', 
        the specified covariate, and 'spikes_per_second'.
    area : str
        The brain area to filter the dataset by.
    covariate : str
        The covariate to analyze (e.g., 'temporal_frequency', 'orientation', etc.).
    plot : bool
        If True, generate plots for the fitted models.

    Returns:
    pandas.DataFrame
        A DataFrame where each row corresponds to a unit in the specified area, and columns 
        contain the results of the model fitting and statistical tests (e.g., R², p-values, slope).
    """
    filtered_dataset = dataset[dataset['area'] == area][['unit_id', covariate, 'spikes_per_second']]
    results = {}
    for unit in filtered_dataset['unit_id'].unique():
        unit_data = filtered_dataset[filtered_dataset['unit_id'] == unit]
        results[unit] = fit_and_test_models(unit_data[covariate].values, unit_data['spikes_per_second'].values, unit, area, covariate, plot)
    return pd.DataFrame(results).T

def summarise_area(dataset, area, covariate, alpha=0.05, plot=False):
    """
    Summarises statistical analysis results for a specific brain area.

    This function analyzes the relationship between a covariate and neural activity
    in a specified brain area, returning a summary of significant findings.

    Args:
        dataset (pd.DataFrame): The dataset containing neural activity data.
        area (str): The name of the brain area to analyze.
        covariate (str): The covariate to test against neural activity.
        alpha (float, optional): The significance level for hypothesis testing. Defaults to 0.05.
        plot (bool, optional): Whether to generate plots during analysis. Defaults to False.

    Returns:
        dict: A dictionary containing the following keys:
            - 'n_units' (int): Total number of units analyzed.
            - 'n_linear_significant' (int): Number of units with significant linear relationships.
            - 'n_positive' (int): Number of units with significant positive slopes.
            - 'n_negative' (int): Number of units with significant negative slopes.
            - 'n_sin_significant' (int): Number of units with significant sinusoidal relationships.
    """
    df = analyse_area(dataset, area, covariate, plot)
    n_units = len(df)
    n_linear_significant = len(df[df['linear_p_value'] < alpha])
    n_positive = len(df[(df['slope_direction'] == 'positive') & (df['linear_p_value'] < alpha)])
    n_negative = n_linear_significant - n_positive
    n_sin_significant = len(df[df['sin_p_value'] < alpha])
    return {
        'n_units': n_units,
        'n_linear_significant': n_linear_significant,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_sin_significant': n_sin_significant
    }

