# GNR: Genetic-Embedded Nuclear Reaction Optimization for Gene Selection

This repository contains the Python implementation of the GNR algorithm proposed in our paper, along with the benchmark microarray datasets used for evaluation.

## Contents
- `GNR.py`: Full code for preprocessing, F-score filtering, and hybrid optimization using NRO with embedded genetic crossover.
- `datasets/`: Contains six publicly available microarray cancer datasets in ARFF format used in the experiments.

## How to Run
1. Make sure you have the following Python packages installed:
   - `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`

2. Verify that the six `.arff` datasets are located in the `datasets/` folder (included in this repository), or update the `data_files` paths in the script accordingly.

3. Run the script with:
   ```bash
   python GNR.py
   ```

## Notes
- The datasets included are standard benchmark datasets originally published in peer-reviewed studies and are commonly used in gene selection and classification research.
- This repository is intended to support reproducibility and further research on hybrid metaheuristic feature selection algorithms.
