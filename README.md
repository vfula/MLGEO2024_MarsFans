# MLGEO2024 Mars Fans Project

A project for ESS 469/569 in Autumn 2024 by Trent Thomas and Veronica Fula.

## Overview

This repository contains instructions and code for analyzing images of alluvial fans and deltas on Mars's surface, with a primary focus on the Eberswalde crater. The project involves training and applying a self-supervised deep learning algorithm to segment images of these fans.

## Directory Structure

- `download_data.ipynb`: Instructions for how to download the dataset we used at multiple different levels of processing.
- `generate_clean_data.ipynb`: Data pipeline for going from raw data (lvl 1) to cleaned and clipped data (lvl 2).
- `generate_ai-ready_data.ipynb`: Data pipeline for going from cleand and clipped data (lvl 2) to AI ready data (lvl 3).
- `visualize_ai-ready_data.ipynb`: Visualize the AI ready data (level 3).
- `analysis_exploratory-dimreduce.ipynb`: An exploratory analysis and dimensionality reduction of the data.

## Getting Started

To get started, follow these steps:

1. **Download and Prepare Data**:

   - Go to the `download_data.ipynb` notebook for instructions on downloading and viewing the training data.
   - There are 3 levels of data available for download, or you can start from the raw data and reproduce our entire analysis.
   - We recommend creating the following directories for the data: `data_lvl1_raw1`, `data_lvl2_cleaned-clipped`, and `data_lvl3_ai-ready`.

2. **Visualize the AI Ready Data**:

   - Ensure you have the level 3 data, either by downloading directly or following our data pipeline.
   - Run the `visualize_ai-ready_data.ipynb` notebook to view the training images.

3. **Exploratory Data Analysis and Demonsionality Reduction**

   - View the `analysis_exploratory-dimreduce.ipynb` notebook to understand the basic structure of the data and see some simple analysis.

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
