# MLGEO2024 Mars Fans Project

A project for ESS 469/569 in Autumn 2024 by Trent Thomas and Veronica Fula.

## Overview

This repository contains instructions and code for analyzing images of alluvial fans and deltas on Mars's surface, with a primary focus on the Eberswalde crater. The project involves training and applying a self-supervised deep learning algorithm to segment images of these fans.

## Directory Structure

- `data/`: contains all of the data for this project, classified into `raw`, `clean`, and `AI_ready`. Note that these folders are empty to start because the data is not hosted on github.
- `notebooks/`: contains all of the jupyter notebooks needed to download, process, and analyze the data for this project, including
    - `Download_Data.ipynb`: downloads the raw, clean, or AI ready data.
    - `Data_Cleaning.ipynb`: data pipeline for going from raw to clean data.
    - `Prepare_AI_Ready_Data.ipynb`: data pipeline for going from clean to AI ready data.
    - `EDA.ipynb`: exploratory data analysis.
    - `Dimensionality_Reduction.ipynb`: a sample analysis to demonstrate dimensionality reduction and other simple ML analysis.

## Getting Started

To get started, follow these steps:

1. **Download and Prepare Data**:

   - Go to the `Download_Data.ipynb` notebook for instructions on downloading and viewing the training data.
   - There are 3 levels of data available for download, or you can start from the raw data and reproduce our entire analysis.

2. **Explore the AI Ready Data**:

   - Ensure you have the AI ready data, either by downloading directly or following our data pipeline.
   - Run the `EDA.ipynb` notebook to view the training images and see a simple statistical analysis.

## Requirements

- Python 3.8+
- Jupyter Notebook
- TensorFlow
- rioxarray
- scikit-image
- matplotlib
- geopandas
- pyproj
- dask

## License

This project is licensed under the MIT License. See the LICENSE file for details.
