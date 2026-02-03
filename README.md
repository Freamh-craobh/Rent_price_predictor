Dublin Rent Price Prediction
Overview
This project analyzes Dublin rental market data from 2008-2021 sourced from the Central Statistics Office (CSO) and builds machine learning models to predict rent prices. The analysis includes data cleaning, exploratory visualization, geospatial mapping, and predictive modeling.

Dataset
Source: Central Statistics Office (CSO) / Residential Tenancies Board (RTB)
Time Period: 2008-2021
Geographic Focus: Dublin postal regions (Dublin 1-24)
Features: Year, Location, Number of Bedrooms, Property Type, Average Monthly Rent (VALUE)

Project Structure
1. Data Loading & Cleaning
Concatenates two CSV files (drp1.csv, drp2.csv)
Removes irrelevant columns (STATISTIC Label, UNIT)
Handles duplicates and missing values
Standardizes location names to Dublin postal codes
2. Exploratory Data Analysis
Statistical summary showing average rent of €1,349.78 (2008-2021)
Analyzes bedroom categories ("1 to 2 bed" vs individual bed counts)
Decision: Groups "1 to 2 bed" with "2 bed"; drops "1 to 3 bed" rows
3. Feature Engineering
One-hot encoding: Property Type and Location
Label encoding: Number of Bedrooms (1, 2, 3, 4)
Binary encoding: Used for Random Forest model
4. Geospatial Visualization
Creates animated choropleth maps showing rent price trends by Dublin postal region (2008-2021)
Reveals effects of 2008 financial crisis and COVID-19 pandemic on rental market
5. Predictive Modeling
Linear Regression with PCA
Applied PCA for dimensionality reduction (35 → 8 components for 95% variance)
R² Score: ~1.0 (though with high prediction errors for outliers)
High MSE indicates poor generalization

Random Forest Regressor
Binary-encoded features for Location and Property Type
100 estimators, 80/20 train-test split
Hyperparameter tuning via GridSearchCV
Better performance on visual inspection vs Linear Regression

Key Findings
Market Trends: Average rents dropped ~23% from 2008-2011 (financial crisis), then recovered with slower growth post-2020 (COVID-19)
Location Impact: Dublin 4 shows consistently higher rents; Dublin 1 displays high volatility
Bedroom Correlation: Strong ordinal relationship between number of bedrooms and price
Encoding Decision: Label encoding bedrooms preserved ordinal patterns better than one-hot encoding
