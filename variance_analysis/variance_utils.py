import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_unit_variance(dataset, unit, frequency):
    """
    Compute the variance of spikes per second for a specific unit in the dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    unit (int): The unit ID for which the variance is to be computed.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    float: The variance of spikes per second for the specified unit.
    """
    filtered_dataset = dataset.loc[dataset['unit_id'] == unit]
    variance = filtered_dataset.groupby(['orientation', frequency]).mean(numeric_only=True).reset_index().drop(columns=['stimulus_presentation_id'])['spikes_per_second'].std()**2
    return variance

def compute_area_variance(dataset, area, frequency):
    """
    Compute the variance of spikes per second for all units in a specific brain area.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    area (str): The brain area for which the variance is to be computed.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    float: The variance of spikes per second for the specified brain area.
    """
    filtered_dataset = dataset.loc[dataset['area'] == area]
    variance = filtered_dataset.groupby(['unit_id', 'orientation', frequency]).mean(numeric_only=True).reset_index().drop(columns=['stimulus_presentation_id'])['spikes_per_second'].std()**2
    return variance

def compute_dataset_variance(dataset, frequency):
    """
    Compute the variance of spikes per second for all units in the dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    dict: A dictionary where the keys are unit IDs and the values are the variances of spikes per second for the corresponding units.
    """
    units = dataset['unit_id'].unique()
    variances = {}
    for unit in units:
        variance = compute_unit_variance(dataset, unit, frequency)
        variances[unit] = variance
    return variances

def compute_dataset_variance_area(dataset, frequency):
    """
    Compute the variance of spikes per second for all units in each brain area.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    dict: A dictionary where the keys are brain areas and the values are the variances of spikes per second for the corresponding areas.
    """
    areas = dataset['area'].unique()
    variances = {}
    for area in areas:
        variance = compute_area_variance(dataset, area, frequency)
        variances[area] = variance
    return variances

def histogram_variances(variances):
    """
    Plot a histogram of variances for units.

    Parameters:
    variances (dict): A dictionary where keys are unit IDs and values are the variances of spikes per second.

    Returns:
    None: Displays a histogram plot of the variances.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(list(variances.values()), bins=30, ax=ax)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Variances')
    plt.show()

def bar_plot_variances(variances):
    """
    Plot a bar plot of variances for units.

    Parameters:
    variances (dict): A dictionary where keys are unit IDs and values are the variances of spikes per second.

    Returns:
    None: Displays a bar plot of the variances.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(variances.keys()), y=list(variances.values()), ax=ax)
    ax.set_xlabel('Area')
    ax.set_ylabel('Variance')
    ax.set_title('Bar Plot of Variances')
    plt.xticks(rotation=90)
    plt.show()

def top_variances(variances, percentage=0.1):
    """
    Extract the top percentage of units with the highest variances.

    Parameters:
    variances (dict): A dictionary where keys are unit IDs and values are their variances.
    percentage (float): The percentage of top variances to extract (default is 0.1, i.e., top 10%).

    Returns:
    dict: A dictionary containing the top percentage of units with the highest variances.
          Keys are unit IDs, and values are their variances.
    """
    sorted_variances = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    top_percent = sorted_variances[:int(len(sorted_variances) * percentage)]
    return dict(top_percent)

def get_unit_area(dataset, unit_id):
    """
    Retrieve the brain area associated with a specific unit.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units and their associated areas.
    unit_id (int): The unit ID for which the brain area is to be retrieved.

    Returns:
    str or None: The brain area associated with the unit, or None if the unit is not found in the dataset.
    """
    unit = dataset.loc[dataset['unit_id'] == unit_id]
    if unit.empty:
        return None
    else:
        return unit.iloc[0]['area']

def get_area_frequency(dataset, unit_list):
    """
    Compute the frequency of units in each brain area.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units and their associated areas.
    unit_list (iterable): A list or iterable of unit IDs for which the brain area frequencies are to be computed.

    Returns:
    dict: A dictionary where keys are brain areas and values are the count of units in each area.
    """
    areas = {}
    for unit_id in unit_list:
        area = get_unit_area(dataset, unit_id)
        if area is not None:
            if areas.get(area) is None:
                areas[area] = 0
            areas[area] += 1
    return areas

def bar_plot_areas(areas):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(areas.keys()), y=list(areas.values()), ax=ax)
    ax.set_xlabel('Area')
    ax.set_ylabel('# of high-variance units')
    ax.set_title('Bar Plot of Areas')
    plt.xticks(rotation=90)
    plt.show()

def plot_unit_spikes_per_second(dataset, unit, frequency):
    """
    Plot the spikes per second for a specific unit in the dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    unit (int): The unit ID for which the spikes per second are to be plotted.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    None: Displays a scatter plot of spikes per second for the specified unit.
    """
    filtered_dataset = dataset.loc[dataset['unit_id'] == unit]
    spikes_dataset = filtered_dataset.groupby(['orientation', frequency]).mean(numeric_only=True).reset_index().drop(columns=['stimulus_presentation_id'])

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=spikes_dataset[frequency], y=spikes_dataset['orientation'], hue=spikes_dataset['spikes_per_second'], palette='Reds', size=spikes_dataset['spikes_per_second'], sizes=(20, 200))
    plt.title(f'Spikes per second for unit {unit} in area {get_unit_area(dataset, unit)}')
    plt.legend(bbox_to_anchor=(1.01, 1), title='Spikes per second', loc='upper left')
    plt.show()

def plot_area_spikes_per_second(dataset, area, frequency):
    """
    Plot the spikes per second for all units in a specific brain area.

    Parameters:
    dataset (pd.DataFrame): The dataset containing information about units, orientations, frequencies, and spikes per second.
    area (str): The brain area for which the spikes per second are to be plotted.
    frequency (str): The column name representing the frequency (e.g., 'temporal_frequency').

    Returns:
    None: Displays a scatter plot of spikes per second for all units in the specified area.
    """
    filtered_dataset = dataset.loc[dataset['area'] == area]
    spikes_dataset = filtered_dataset.groupby(['orientation', frequency]).mean(numeric_only=True).reset_index().drop(columns=['stimulus_presentation_id', 'unit_id'])

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=spikes_dataset[frequency], y=spikes_dataset['orientation'], hue=spikes_dataset['spikes_per_second'], palette='Reds', size=spikes_dataset['spikes_per_second'], sizes=(20, 200))
    plt.title(f'Spikes per second for units in area {area}')
    plt.legend(bbox_to_anchor=(1.01, 1), title='Spikes per second', loc='upper left')
    plt.show()