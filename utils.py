import os

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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