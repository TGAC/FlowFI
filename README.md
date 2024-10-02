# FlowFI

FlowFI (Flow cytometry Feature Importance) is a python-based, graphical interface to enable experimentalists to perform online data driven feature importance analysis of flow cytometry spectral and possibly also imaging features for gating. The software uses efficient spectral methods for feature importance analysis with parallel processing to enable analysis of large numbers of live samples for refinement of the gating approach at the bench. 

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
1. Select your .fcs file using the browser (filtering out samples e.g. using pre-gating before saving).

![gui](https://github.com/jameswilsenach/FlowFI/blob/main/gui.png?raw=true)

2. Run FlowFI by pressing the execute button.

![gui_run](https://github.com/jameswilsenach/FlowFI/blob/main/gui_run.png?raw=true)

3. Features appear in order from most to least important and are colour coded by type with the bar representing relative imprtance from 0-1.

4. Select or deselect the checkboxes to show or hide features in the output.

5. Use Execute again to rerun the analysis on the subset of features that are currently select.

![gui_refine](https://github.com/jameswilsenach/FlowFI/blob/main/gui_refine.png?raw=true)  

6. Save your output in .csv format using the File -> Save option.

7. See the Help section for additional guidelines.

# Understanding the Feature/Importance Window:
Features are colour-coded in FlowFI to correspond to their type (e.g. Imaging, red, blue, etc.). After execution, each feature is assigned an importance score. This is shown in the bar to the right with a large bar indicating high importance. By default, features are ordered by their estimated importance but this can be changed using the dropdown "Sort by:" menu. Features are also assigned a cluster (coloured border) and features that are the most central (i.e. typical of a collection of features) are shown in bold and underlined.


# FlowFI Save Files:
FlowFI saves the output of the analysis in a CSV file (excel-readable table) with columns:\
feature - name of the corresponding feature in the original fcs file\
ri - Relative Importance, this is the importance of the feature from 0-1 with 1 being the highest importance\
ls - Laplace Score - This is the raw basis of the importance which is used for algorithmic purposes\
membership - numerical id of the cluster this feature belongs to\
centrality - numerical score from 0-1 with 1 corresponding to the feature with the highest centrality



# Creating a FlowFI Executable from Source
These instructions assume a working version of Python >=3.10 that is accessible from the command line. In order to prevent the executable from becoming too large, it is best to make the executable in a fresh virtual environment. To do this in Linux, navigate to the FlowIO directory in the command line and use the following commands:
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
[1] He, X., Cai, D., & Niyogi, P. (2005). Laplacian score for feature selection. Advances in neural information processing systems, 18.\
[2] Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 1-12.
