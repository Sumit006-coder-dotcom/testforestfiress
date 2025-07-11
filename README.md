# testforestfires

#  TesForestFires - Fire Weather Index Prediction

A machine learning project that predicts the **Fire Weather Index (FWI)** based on weather and environmental factors using Lasso Regression.

## Project Overview

This project focuses on predicting fire risk using various meteorological and forest conditions. The prediction is shown on a simple web interface built using Flask and HTML. The goal is to demonstrate a real-time machine learning prediction system with a clean UI and reliable backend.

## Features

- Exploratory Data Analysis (EDA)
- Feature Engineering (FE)
- Lasso Regression Model
- Web-based prediction interface
- Deployment-ready structure (pickle model + scaler)
- Based on inputs like Temperature, RH, Wind Speed, Rainfall, FFMC, DMC, ISI, Region, and Fire Class

Model Used: Lasso Regression (sklearn.linear_model.Lasso)

Preprocessing: StandardScaler

Pickle Files: ridge.pkl (model), scaler.pkl (preprocessing)

 Libraries Used

Python

NumPy

Pandas

Scikit-learn

Flask

Jupyter Notebook
