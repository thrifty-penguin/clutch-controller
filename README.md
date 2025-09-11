# Adaptive Clutch Modulator

This project presents a data-driven approach to developing an adaptive clutch engagement controller. It leverages machine learning to analyse real-world driving data, identify distinct clutch engagement profiles, and build a predictive model that selects the optimal clutch modulation strategy based on initial vehicle conditions.

The final output is a simulated real-time controller that can intelligently modulate clutch engagement to balance ride comfort (low jerk) and component longevity (low wear).

## Table of Contents

- [Adaptive Clutch Modulator](#adaptive-clutch-modulator)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Methodology](#methodology)
  - [Technical Stack](#technical-stack)
  - [How to Run](#how-to-run)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Execution](#execution)
  - [Contribution](#contribution)

## Project Overview

Clutch control is critical for ensuring smooth gear shifts and minimizing wear on the clutch system. A one-size-fits-all control strategy is often suboptimal, as the ideal engagement profile can vary significantly based on factors like engine speed, torque, and vehicle speed.

This project tackles that challenge by:
1.  **Analyzing Real-World Data**: Ingesting and processing `.mat` files from AMT vehicle tests. These profiles are pre-tuned and ready for deployment in manual vehicles. The AMT used here is the Daimler Truck G90AMT and the manual transmission it's used to model is the Daimler Truck G131.
2.  **Identifying Engagement Profiles**: Isolating hundreds of individual clutch engagement events and using cubic splines to standardise them.
3.  **Clustering Profiles**: Using K-Means clustering on statistical and shape-based features of the splines to discover distinct, recurring engagement patterns (e.g., "fast and aggressive," "slow and smooth").
4.  **Training a Predictive Model**: Building an XGBoost classifier that predicts which engagement cluster is most appropriate based on the vehicle's state *at the start* of the clutch event.
5.  **Defining "Ideal" Curves**: For each cluster, identifying the single best real-world event that resulted in the lowest combined cost of driveline jerk and clutch wear.
6.  **Simulating a Controller**: Developing a real-time controller that uses the driver's pedal input to select a predicted "ideal" curve, effectively translating driver intent into an optimised, automated clutch action.

## Methodology

The pipeline is executed in a series of modular steps:

1.  **Data Ingestion**: MATLAB `.mat` files are converted into a more accessible `.csv` format.
2.  **Time Alignment**: Data from various sensors, recorded at different frequencies, is aligned to a common time vector using linear interpolation.
3.  **Feature Calculation**: Key performance metrics like slip power and vehicle speed are calculated from raw sensor data.
4.  **Event Segmentation**: The continuous time-series data is segmented into individual clutch engagement events.
5.  **Spline Fitting**: Each event's clutch position curve is fitted to a cubic spline to create a fixed-length vector representation, making them comparable.
6.  **Unsupervised Clustering**: The splines are clustered to group similar engagement styles.
7.  **Supervised Classification**: An XGBoost model is trained to predict the optimal cluster ID using the initial conditions of the engagement event as features. Hyperparameters are tuned using Optuna.
8.  **Cost Analysis**: Each event is assigned a cost based on a weighted score of estimated driveline jerk and clutch wear energy. The "ideal" curve for each cluster is the one with the lowest cost.
9.  **Real-Time Simulation**: A final script simulates the controller's logic, using a sample manual driving dataset to show how it would modulate the clutch in response to pedal input.

## Technical Stack

*   **Core Libraries**: Python 3.12, Pandas, NumPy, SciPy
*   **Machine Learning**: Scikit-learn, XGBoost, Optuna
*   **File Handling**: H5py (for `.mat` files), Joblib (for saving models/scalers)
*   **Visualization**: Matplotlib, SciencePlots (optional)
*   **Environment**: Jupyter Notebook *or* Individual Python Scipts

## How to Run

### Prerequisites

*   Python 3.10+
*   Ensure all required Python packages are installed:
    ```
    pip install pandas numpy scipy scikit-learn xgboost optuna h5py joblib matplotlib scienceplots
    ```

### Installation

1.  Clone this repository to your local machine.
2.  Create the directory structure as shown in the [File Structure](#file-structure) section.
3.  Place your raw `.mat` test data files in the `data/amt_raw/` directory.
4.  Place a `.csv` file with manual driving data (for the final simulation) in the `data/man_raw/` directory. Ensure it has the columns specified in the final notebook cell.

### Execution

1.  Open the `clutch-controller.ipynb` notebook in a Jupyter environment.
2.  Execute the cells sequentially from top to bottom.
3.  The notebook will:
    *   Process the raw data.
    *   Train the machine learning model.
    *   Save the model and curve artifacts to the `artifacts/` directory.
    *   Generate and display plots showing the clustered curves and the final controller simulation.

## Contribution

This project was developed as an engineering internship task. While it is a complete proof-of-concept, contributions and suggestions are welcome. Potential areas for improvement include:

*   **Enhanced Error Handling**: Adding more robust checks for file I/O and data integrity.
*   **Unit Testing**: Implementing tests for key functions to ensure reliability.
*   **Configuration Management**: Moving more "magic numbers" from the code into the `Config` class.
*   **Code Refactoring**: Encapsulating the main execution logic into functions for better reusability.
```
