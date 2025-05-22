# Neural Responses to Visual Stimuli: The Role of Temporal Frequency, Spatial Frequency, and Orientation
Group project for the BAI course 30563 - Mathematical Modelling for Neuroscience

## Group members
Edoardo Ghirardo, Giuseppe Iannone, Francois Maurice Marie Hoche, Filippo Antonio Ronzino, Elisa Tofanelli.

## Structure
- `00_session_analysis.ipynb`: Initial session analysis notebook
- `utils.py`: Common utility functions for the entire project
- `01_data_analysis/`: Contains notebooks and utilities for data processing
  - `data_analysis_drifting.ipynb`: Analysis of drifting grating responses
  - `data_analysis_static.ipynb`: Analysis of static grating responses
  - `data_analysis_utils.py`: Shared functions for data analysis
- `02_variance_analysis/`: Contains code for variance analysis
  - `drifting_variance_analysis.ipynb`: Variance analysis for drifting gratings
  - `static_variance_analysis.ipynb`: Variance analysis for static gratings
  - `variance_utils.py`: Utilities specific to variance calculations
- `03_hypothesis_testing/`: Statistical testing implementations
  - `testing.ipynb`: Hypothesis tests on the neuronal data
  - `regression_utils.py`: Regression analysis utilities
- `04_clustering_classifier/`: Machine learning approaches for neuron classification
- `allendata/`: Contains Allen Institute Brain Observatory dataset files
- `ourdata/`: Contains our processed datasets
