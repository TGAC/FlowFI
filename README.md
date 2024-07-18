## FlowFI

FlowFI (Flow cytometry Feature Importance) is a python-based, graphically enabled interface to enable experimentalists to perform online data driven feature importance analysis of flow cytometry spectral and possibly also imaging features for gating. The software uses efficient spectral methods for feature importance analysis with parallel processing to enable analysis of large numbers of live samples for refinement of the gating approach at the bench. 

The software is capable of analysing data from any generic .fcs file but was tested using data generated with the BD FACSDiscover™ S8 Cell Sorter from BD Biosciences that can provide a range of spectral and imaging features. FlowFI does not perform or suggest a gating strategy, but instead ranks features by how much of the variance in the samples they account for. This is performed using robust spectral methods based on Laplace scoring [1].

FlowFI allows for a subset of features (e.g. imaging vs specific red, violet or blue features) to be analysed, allowing for results to be iteratively refined based on the subset of interest.

# Installation
To install FlowFI in Python 3.10 or above, download the repository and navigate to the FlowFI directory. Then use the following command in your command line:

```
pip install -r requirements.txt
```

# Run FlowFI
While in the FlowFI directory, run the following in the commandline:
```
python3 main.py
```

![alt text](https://github.com/jameswilsenach/FlowFI/blob/main/gui.png?raw=true)

