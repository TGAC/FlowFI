# FlowFI

FlowFI (Flow cytometry Feature Importance) is a python-based, graphically enabled interface to enable experimentalists to perform online data driven feature importance analysis of flow cytometry spectral and possibly also imaging features for gating. The software uses efficient spectral methods for feature importance analysis with parallel processing to enable analysis of large numbers of live samples for refinement of the gating approach at the bench. 

The software is capable of analysing data from any generic .fcs file but was tested using data generated with the BD FACSDiscover™ S8 Cell Sorter from BD Biosciences that can provide a range of spectral and imaging features. FlowFI does not perform or suggest a gating strategy, but instead ranks features by how much of the variance in the samples they account for. This is performed using robust spectral methods based on Laplace scoring [1].

FlowFI allows for a subset of features (e.g. imaging vs specific red, violet or blue features) to be analysed, allowing for results to be iteratively refined based on the subset of interest.

## Installation
To install FlowFI in Python 3.10 or above, download the repository and navigate to the FlowFI directory. Then use the following command in your command line:

```
pip install -r requirements.txt
```

## Run FlowFI
While in the FlowFI directory, run the following in the commandline:
```
python3 main.py
```

FlowFI uses [FlowIO](https://github.com/whitews/FlowIO) to load .fcs files and PyQt5 to implement a Graphical User Interface (GUI) for handling of input and output files and visualisation of results.

## Using FlowFI

![gui](https://github.com/jameswilsenach/FlowFI/blob/main/gui.png?raw=true)

1. Select your .fcs file using the browser (filtering out samples e.g. using pre-gating before saving).
2. Run FlowFI by pressing the execute button.
3. Features appear in order from most to least important and are colour coded by type.
4. Select or deselect the checkboxes to show or hide features in the output.
5. Use Execute again to rerun the analysis on the subset of features that are currently select.
6. Save your output in .csv format using the File -> Save option.
7. See the Help section for additional guidelines.

## Creating a FlowIO Executable from source
In order to prevent the executable from becoming too large, it is best to make the executable in a fresh virtual environment. To do this in Linux, navigate to the FlowIO directory in the command line and use the following commands:
```
pip install virtualenv
virtualenv makeenv
source makeenv/bin/activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --onefile --windowed main.py -n flowfeatures
```
the executable is found inside of the ./dist directory.

In Windows, this would be
```
pip install virtualenv
virtualenv makeenv
makeenv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --onefile --windowed main.py -n flowfeatures.exe
```



# References
[1] He, X., Cai, D., & Niyogi, P. (2005). Laplacian score for feature selection. Advances in neural information processing systems, 18.
