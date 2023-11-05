#### ===================================== ####
     Bachelor Thesis Code - Fabian Bermana
#### ===================================== ####

This document describes the content of this code base and datasets. All code that produce output have been stored in Jupyter notebooks. Additional script files contain code required for these notebooks to run.


### Jupyter Notebooks ###

- Data.ipynb: cleaning and combining data

- Extension.ipynb: most extensions to the main paper

- Neural Network.ipynb: evaluation of neural network models

- Replication.ipynb: replication of results from Zhang et al (2020)

- Results.ipynb: creation of tables and figures


### Data Files ###

- data_return.csv: CRSP value-weighted index returns

- data_stocks.csv: main dataset, cleaned and formatted

- GoyalData_2021.csv: raw data from website of Amit Goyal


### Python Script Files ###

- forecast_evaluation.py: functions to calculate performance metrics

- forecast_methods.py: functions to produce forecasts

- forecast_models.py: classes for econometric forecast models

- ml_models.py: classes for machine learning forecast models


### Folders ###

- forecast_results: predictions made using econometric forecast models

- MATLAB: Matlab script files, required for WALS

- ml_results: predictions made using machine learning forecast models

- nn_results: predictions made using neural network models






