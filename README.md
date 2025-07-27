# GNR: Genetic-Embedded Nuclear Reaction Optimization for Gene Selection

This repository contains the Python implementation of the GNR algorithm proposed in our paper.

## Contents
- `GNR.py`: Full code for preprocessing, F-score filtering, and hybrid optimization using NRO with embedded genetic crossover.

## How to Run
1. Make sure you have the following Python packages installed:
   - numpy, pandas, scikit-learn, scipy, tqdm

2. Place the six `.arff` datasets in a local folder and update the `data_files` list in the script with their correct paths.

3. Run the script with:
   ```bash
   python GNR.py
