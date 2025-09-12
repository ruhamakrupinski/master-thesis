# Master Thesis: Poverty Measurement and Targeting (PMT) Model

This repository contains the implementation of a two-stage Poverty Measurement and Targeting (PMT) model for the Philippines, focusing on the National Capital Region (NCR). The project combines data from multiple sources including the Family Income and Expenditure Survey (FIES), Labor Force Survey (LFS), Census data, and Demographic and Health Survey (DHS) to develop a comprehensive poverty targeting system.

## Overview

The PMT model uses a two-stage approach:
1. **Linear Regression Stage**: Predicts household income using observable characteristics
2. **Logistic Regression Stage**: Refines poverty classification using a threshold-based approach

The model incorporates three types of indicators:
- **Household Indicators**: Building characteristics, assets, utilities
- **Individual Indicators**: Demographics, education, occupation
- **Community Indicators**: Infrastructure, accessibility, services

## Project Structure

### Core Files

- **`PMTModel.py`**: Main PMT model implementation with two-stage prediction pipeline
- **`Census_mapping.py`**: Utility functions for data cleaning and preprocessing
- **`mapping.py`**: Data cleaning utilities for handling missing values and categorical data
- **`education_recode.py`**: Education code mapping and standardization functions

### Data Processing Notebooks

- **`FIES_18_OCC.ipynb`**: FIES & LFS 2018 data preparation and cleaning
- **`FIES_21_OCC.ipynb`**: FIES & LFS 2021 data preparation and cleaning  
- **`PMT2_OCC.ipynb`**: Main PMT model training and evaluation
- **`Community_ind_prep.ipynb`**: Community indicators preparation from census data
- **`MAP.ipynb`**: Geographic mapping and visualization using shapefiles

### Mapping Dictionaries

- **`Mapping Dictionary for Thesis.py`**: Comprehensive mapping dictionaries for categorical variables
- **`Mapping Dictionary for Thesis_2021.py`**: Updated mappings for 2021 data

## Setup Instructions

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd master-thesis
   ```

2. **Initialize the project with uv**:
   ```bash
   uv init
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   uv shell
   ```

### Data Setup

**Note**: The data files are not included in this repository due to size constraints and privacy considerations. You will need to obtain the following datasets separately:

#### Required Data Sources

1. **Census Data** (`data/Census/`):
   - 2020 Census of Population and Housing (CPH) PUF
   - Form 3 (Sample Household Questionnaire) data
   - Form 5 (Barangay Schedule Questionnaire) data

2. **FIES & LFS Data** (`data/FIES&LFS/`):
   - FIES-LFS PUF 2018 merged dataset
   - FIES-LFS PUF 2021 merged dataset

3. **DHS Data** (`data/DHS/`):
   - Philippines DHS 2017 and 2022 datasets
   - Various modules (BR, FW, GC, GE, GR, HR, IR, KR, NR, PR)

4. **Shape Files** (`data/Shape files/`):
   - Philippines PSGC Barangay shapefiles
   - Geographic boundary data for mapping

#### Data Organization

Place the data files in the following structure:
```
data/
├── Census/
│   └── PHL PSA CPH 2020 PUF/
├── DHS/
│   ├── PH_2017_DHS_11032024_53_220775.zip
│   └── PH_2022_DHS_11032024_52_220775/
├── FIES&LFS/
│   ├── FIES LFS Merge 2018/
│   └── FIES LFS Merge 2021/
└── Shape files/
    └── Philippines PSGC Bgy Sub Muns/
```

## Dependencies

The project uses the following key dependencies:

- **`pandas`** (≥2.3.2): Data manipulation and analysis
- **`numpy`**: Numerical computing
- **`scikit-learn`** (≥1.7.2): Machine learning algorithms and preprocessing
- **`statsmodels`** (≥0.14.5): Statistical modeling and OLS regression
- **`xgboost`**: Gradient boosting for regression and classification
- **`geopandas`** (≥1.1.1): Geospatial data analysis
- **`matplotlib`** (≥3.10.6): Data visualization
- **`seaborn`**: Statistical data visualization
- **`pyreadstat`** (≥1.3.1): Reading statistical data files (DHS, FIES)
- **`ipykernel`** (≥6.30.1): Jupyter notebook support

## Usage

### Running the Analysis

1. **Prepare the data** by running the data processing notebooks in order:
   ```bash
   # Start Jupyter Lab
   uv run jupyter lab
   ```

2. **Execute notebooks sequentially**:
   - `FIES_18_OCC.ipynb` - Process 2018 FIES/LFS data
   - `FIES_21_OCC.ipynb` - Process 2021 FIES/LFS data
   - `Community_ind_prep.ipynb` - Prepare community indicators
   - `PMT2_OCC.ipynb` - Train and evaluate PMT model
   - `MAP.ipynb` - Generate geographic visualizations

## Methodology

### Model Approaches

The project implements and compares three different modeling approaches for poverty targeting:

#### 1. Two-Stage PMT (Poverty Measurement and Targeting)

**Stage 1 - Linear Regression (OLS)**:
- Predicts log-transformed per capita income using observable characteristics
- Uses household, individual, and community indicators (75+ variables)
- Generates confidence intervals for income predictions
- Applies 40th percentile threshold for initial poverty classification

**Stage 2 - Logistic Regression**:
- Refines poverty classification using predicted income thresholds
- Applies additional filtering based on probability scores

#### 2. XGBoost Machine Learning Model

- **Regression Component**: Predicts log per capita income using gradient boosting
- **Classification Component**: Direct poverty classification using probability thresholds
- **Features**: Native categorical support, hyperparameter tuning

#### 3. Weighted Models

All models include weighted versions using survey weights to account for sampling design and improve population-level estimates.


### Technical Implementation

#### Data Preprocessing Pipeline
- **Missing Value Handling**: Custom replacement strategies for "Not Reported" values
- **Categorical Encoding**: One-hot encoding for categorical variables, native categorical support for XGBoost
- **Feature Engineering**: Age group indicators, education level buckets, occupation categories
- **Data Standardization**: StandardScaler for numerical features in logistic regression
- **Survey Weighting**: Normalized weights for population-level inference

#### Model Architecture Details
- **OLS Regression**: Statsmodels implementation with formula-based specification
- **Logistic Regression**: Scikit-learn with SAGA solver, 100,000 max iterations
- **XGBoost**: Native categorical support, hyperparameter tuning via cross-validation
- **Threshold Selection**: Precision-recall curve optimization for classification thresholds

#### Performance Optimization
- **Cross-Validation**: Stratified k-fold for robust model evaluation
- **Hyperparameter Tuning**: Grid search and Bayesian optimization for XGBoost
- **Memory Management**: Efficient data types and chunked processing for large datasets
- **Parallel Processing**: Multi-threading support for model training and evaluation


## File Descriptions

### Core Implementation
- **`PMTModel.py`**: Main PMT model class implementing two-stage approach (OLS + Logistic Regression)
- **`Census_mapping.py`**: Data cleaning utilities for census data preprocessing
- **`mapping.py`**: General data cleaning and preprocessing functions for missing values
- **`education_recode.py`**: Education code standardization and mapping from PSCED/CBMS to custom scheme

### Data Processing Notebooks
- **`FIES_18_OCC.ipynb`**: Complete 2018 FIES/LFS data preparation pipeline with variable construction
- **`FIES_21_OCC.ipynb`**: Complete 2021 FIES/LFS data preparation pipeline with variable construction
- **`Community_ind_prep.ipynb`**: Community indicators preparation from census data for geographic analysis
- **`PMT2_OCC.ipynb`**: Main analysis notebook containing all model training, evaluation, and comparison
- **`MAP.ipynb`**: Geographic mapping and visualization using shapefiles and survey data

### Mapping Dictionaries
- **`Mapping Dictionary for Thesis.py`**: Comprehensive mapping dictionaries for categorical variables (2018)
- **`Mapping Dictionary for Thesis_2021.py`**: Updated mappings for 2021 data with additional variables

### Configuration
- **`pyproject.toml`**: Project configuration and dependencies specification
- **`uv.lock`**: Locked dependency versions for reproducible environment