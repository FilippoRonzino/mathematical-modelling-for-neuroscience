import pandas as pd


def compute_z_score_per_unit(
    dataset: pd.DataFrame, 
    columns: list = ['spike_mean_baseline', 'spike_std_baseline', 'spikes_per_second'],
) -> pd.DataFrame:
    """
    Compute z-score for each unit relative to its baseline.
    
    Parameters:
    - dataset: DataFrame containing spike data with baseline statistics
    - columns: List of column names needed for z-score calculation
               (baseline mean, baseline std, and current spike rate)
    
    Returns:
    - DataFrame with additional z_score_to_baseline column
    """
    result = dataset.copy()
    
    result['z_score_to_baseline'] = (
        (result['spikes_per_second'] - result['spike_mean_baseline']) / 
        result['spike_std_baseline']
    )
    
    # Set z-score to 0 if std is 0 (no change from baseline)
    zero_std_mask = result['spike_std_baseline'] == 0
    if zero_std_mask.any():
        print(f"Warning: {zero_std_mask.sum()} units have zero standard deviation.")
        result.loc[zero_std_mask, 'z_score_to_baseline'] = 0
        
    return result

def determine_active_unit(unit_data: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Determine if units are active based on their z-score relative to baseline.
    
    Parameters:
    - unit_data: DataFrame containing data for units, with z_score_to_baseline column
                 or data needed to compute it
    - z_threshold: The z-score threshold above which a unit is considered active (default: 2.0)
    
    Returns:
    - DataFrame with additional 'active' column indicating unit activity (1=active, 0=inactive)
    """
    # Make a copy to avoid modifying the original DataFrame
    result = unit_data.copy()
    
    # Compute z-score if not already present
    if 'z_score_to_baseline' not in result.columns:
        if all(col in result.columns for col in ['spike_mean_baseline', 'spike_std_baseline']):
            result = compute_z_score_per_unit(result)
        else:
            raise ValueError("Unit data must contain 'spike_mean_baseline' and 'spike_std_baseline' columns.")
    
    # Mark each unit as active (1) or inactive (0) based on z-score threshold
    result['active'] = (result['z_score_to_baseline'] > z_threshold).astype(int)
    print(f"Unit activity determined with z-threshold: {z_threshold}")
    
    return result

def determine_percentage_active_units_by_area(unit_data: pd.DataFrame):
    """
    Determine the percentage of active units by area.
    
    Parameters:
    - unit_data: DataFrame containing unit data with 'area' and 'active' columns
    
    Returns:
    - DataFrame with percentage of active units for each area
    """
    active_units = unit_data.groupby(['area', 'stimulus_presentation_id'])['active'].mean().reset_index()
    active_units.rename(columns={'active': 'proportion_active_units'}, inplace=True)
    stimulus_info = unit_data[['stimulus_presentation_id', 'temporal_frequency', 'orientation']].drop_duplicates()
    active_units = active_units.merge(stimulus_info, on='stimulus_presentation_id', how='left')

    return active_units
    
