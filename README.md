# Overview

This repository contains code used for analyses in the paper "Constraining inferences of across-region interactions using neural activity perturbations". 

This repository also contains an example notebook (`example.ipynb`) and a partial dataset (pickle files in data) used to run examples of the main analyses used in the paper.

## System Requirements

All software dependencies to run the linear analysis is in the `requirements.txt` folder, which specifies version numbers and operating systems. To run the nonlinear analysis, additionally installing PyTorch and CUDA is required.

Installation should take under 15 minutes. 

## Installation

1) Pull repo

2) Install packages listed in `requirements.txt`

3) Install src package by navigating to root directory of this repo and running `pip install -e .`

4) The pickle files in the data folder (needed for running `example.ipynb`) likely won't automatically download when pulling the repo unless Git LFS is installed. If not, these can be downloaded manually from the repo (navigate to each .pickle file, and click download from dropdown menu), and place in data folder.

## Demo

After following all installation instructions, and if the 3 data files (`.pickle`) are in the data folder, the demo script `example.ipynb` should be able to be run from the top. The entire script should take <5 minutes.  

This demo script outputs the linear regularization sweep for one actual recording session and one simulated session.

## Instructions for Use

All scripts are configured to run if the `session_path` parameter points to source data folders. These source data folders are currently not provided, but a truncated version of these are provided as `.pickle` files for running the demo.
