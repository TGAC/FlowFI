import sys
import csv
import flowio
import numpy as np

import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib

import numpy as np
import traceback

from sklearn_extra.cluster import KMedoids
import leidenalg as la
import igraph as ig

matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QCheckBox, QPushButton, QProgressBar, QLabel, QFileDialog, QScrollArea, QFrame,
                             QAction, QMessageBox,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal,  QTimer
from concurrent.futures import ThreadPoolExecutor

excludedcols = ['Saturated', 'Time', 'Sorted', 'Row', 'Column']
excludedcols += ['Protocol', 'EventLabel', 'Regions0', 'Regions1', 'Regions2',
       'Regions3', 'Gates', 'IndexSort', 'SaturatedChannels', 'PhaseOffset',
       'PlateLocationX', 'PlateLocationY', 'EventNumber0', 'EventNumber1',
       'DeltaTime0', 'DeltaTime1', 'DropId', 'SaturatedChannels1',
       'SaturatedChannels2', 'SpectralEventWidth', 'EventWidthInDrops',
       'SpectralUnmixingFlags', 'WaveformPresent']

BOOT = 1000
CLUSTERS = 10
MEDS = CLUSTERS
BOOTSIZE = 200

# Path to the global CSV file containing feature names

class WorkerThread(QThread):
    progress_update = pyqtSignal(int)
    intermediate_result = pyqtSignal(dict)
    result_ready = pyqtSignal()

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        N = self.data.shape[0]
        self.n=BOOTSIZE
        if N<self.n:
            self.n = N
            self.boots = max([int(N/2),2])
        self.k =int(self.n/3)
        self.mode = 'cosine'
        self.t = 1

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.process_part, i) for i in range(self.boots)]
            for future in futures:
                result = future.result()
                self.intermediate_result.emit(result)
                self.progress = int((len(futures) - len([f for f in futures if not f.done()]))/len(futures)*100)
                # self.progress_update.emit(progress)
        self.result_ready.emit()

    def process_part(self, i):
        ls,medoids,medlabels = self.get_ulscore_parralel()
        return {"value": ls,"i": i,"medoids": medoids,"membership":medlabels}
    
    def get_ulscore_parralel(self):
        n = self.n
        ones = np.ones((n,1))
        sample = np.random.choice(self.data.shape[0],n)
        Xsub = self.data[sample,:]
        Wsub = self.get_similaritymatrix(Xsub)
        Dsub = np.diagflat(np.sum(Wsub,axis=0))
        Lsub = Dsub - Wsub
        LSsub = np.zeros(Xsub.shape[1])
        if CLUSTERS<=self.data.shape[1]/20:
            clusters = CLUSTERS
        else:
            clusters = int(self.data.shape[1]/20)
        model = KMedoids(n_clusters=clusters,method='pam').fit(Xsub.T)
        medoids = model.medoid_indices_
        medlabels = model.labels_
        for r in range(Xsub.shape[1]):
            fsubr = Xsub[:,r].reshape([-1,1])
            neighb_est = ((fsubr.T @ Dsub @ ones).item()/ (ones.T @ Dsub @ ones).item())*ones
            fsubr_est = (fsubr - neighb_est)
            d = (fsubr_est.T @ Dsub @ fsubr_est).item()
            num = (fsubr_est.T @ Lsub @ fsubr_est).item()
            if d > 0 and num>0:
                LSsub[r] = num/d
            elif num==0 and d>0:
                LSsub[r] = 0
            else:
                LSsub[r] = np.inf
        return LSsub,medoids,medlabels

    def get_similaritymatrix(self,X):
        # compute pairwise euclidean distances
        mode = self.mode
        t = self.t
        k = self.k
        n = X.shape[0]
        if mode == 'heat':
            D = pairwise_distances(X)

            Dtop = np.sort(D, axis=1)[:,k+1]
            G = D<=Dtop
            np.fill_diagonal(G,0)
            W = np.zeros([n,n])
            W[G>0] = np.exp(-D[G>0]**2/(2*t**2))

        if mode == 'cosine':

            D = pairwise_distances(X,metric='cosine')

            Dtop = np.sort(D, axis=1)[:,k+1]
            G = D<=Dtop
            np.fill_diagonal(G,0)
            W = np.zeros([n,n])
            W[G>0] = np.abs(1-D[G>0])
            return W


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("File Processor")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        # Input field for filepath
        self.filepath_input = QLineEdit()
        self.filepath_input.setPlaceholderText("Enter file path here")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)

        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.filepath_input)
        self.input_layout.addWidget(self.browse_button)

        # Button to execute the function
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_function)

        self.checkbox_layout = QHBoxLayout()
        self.ftypes = ['UV','V','B','YG','R','ImgB','Imaging','Misc']
        self.colors = ['green','darkviolet','blue','darkgoldenrod','darkred','saddlebrown','teal','black']
        self.clustercolors = ['lightcoral','palegoldenrod','palegreen','lightblue','aquamarine','dimgray','peru','darkseagreen','white','cornflowerblue','green','darkviolet','blue','darkgoldenrod','darkred','saddlebrown','teal','black']
        self.selected_feature_types = self.ftypes
        self.feature_checkboxes = {}
        for i,feature_type in enumerate(self.ftypes):
            checkbox = QCheckBox(feature_type)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_display)
            checkbox.setStyleSheet("color: " + self.colors[i])
            self.feature_checkboxes[feature_type] = checkbox
            self.checkbox_layout.addWidget(checkbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Output display panel
        self.output_panel = QScrollArea()
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()

        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)
        self.output_panel.setWidgetResizable(True)

        # Sorting dropdown box
        self.sort_dropdown = QComboBox()
        self.sort_dropdown.addItem("Sort by: Importance (features that are important to the data structure)")
        self.sort_dropdown.addItem("Sort by: Type (UV, V, etc.)")
        self.sort_dropdown.addItem("Sort by: Cluster (similar features)")
        self.sort_dropdown.addItem("Sort by: Centrality (featuress typical of a cluster)")
        self.sort_dropdown.currentIndexChanged.connect(self.update_display)
        
        self.layout.addLayout(self.checkbox_layout)
        self.layout.addLayout(self.input_layout)
        self.layout.addWidget(self.execute_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.sort_dropdown)
        self.layout.addWidget(QLabel("Feature/Importance:"))
        self.layout.addWidget(self.output_panel)

        self.central_widget.setLayout(self.layout)

        # Menu bar
        self.create_menus()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        # self.update_timer.timeout.connect(self.update_progress)

    def create_menus(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu('File')

        save_action = QAction('Save Output as CSV', self)
        save_action.triggered.connect(self.save_output)
        file_menu.addAction(save_action)

        # Help menu
        help_menu = menu_bar.addMenu('Help')

        readme_action = QAction('README', self)
        readme_action.triggered.connect(self.show_readme)
        help_menu.addAction(readme_action)

    def browse_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)", options=options)
        if filepath:
            self.filepath_input.setText(filepath)

    def execute_function(self):
        filepath = self.filepath_input.text()
        if not filepath:
            QMessageBox.warning(self, "Warning", "Please enter a valid file path.")
            return
        
        self.filepath = filepath
        self.load_features()
        if not hasattr(self,'data'):
            QMessageBox.warning(self, "Warning", "No features found in the FCS file.")
            return

        self.output_layout.removeWidget(self.output_widget)
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)

        # for feature_type in self.ftypes:
        #     if feature_type in self.selected_feature_types:
        #         self.feature_checkboxes[feature_type].setEnabled(False)
        #     else:
        #         self.feature_checkboxes[feature_type].setEnabled(False)


        self.progress_bar.setValue(0)

        self.worker = WorkerThread(self.data)
        self.boots = self.worker.boots
        self.feature_averages = np.zeros((self.data.shape[1],self.boots))
        self.medoids = np.zeros((self.data.shape[1],self.boots))
        self.memberships = np.zeros((self.data.shape[1],self.boots))
        self.finalcluster = False

        
        # self.worker.progress_update.connect(self.update_progress)
        self.worker.intermediate_result.connect(self.add_result)
        self.worker.result_ready.connect(self.finalize_results)
        self.worker.start()

        self.update_timer.setInterval(10000)
        self.update_timer.start()
        QApplication.processEvents()


    def load_features(self):
        try:
            fcdata = flowio.FlowData(self.filepath)
            self.columns = np.array([fcdata.channels[c]['PnN'] for c in fcdata.channels])
            self.data = np.reshape(fcdata.events,[-1,fcdata.channel_count])
            self.cleandata() 

            
        except Exception as e:
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load features from FCS file: {e}")

    def NormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def cleandata(self,norm=True): 
        included = [i for i,c in enumerate(self.columns) if c not in excludedcols]
        self.columns = self.columns[included]
        self.data = self.data[:,included]
        
        # self.data,uind = np.unique(self.data,axis=1,return_index = True)
        # self.columns = self.columns[uind]
        # self.data = self.data[:,uind]

        included = np.var(self.data,axis=0)>0
        nondiverse = [i for i in range(self.data.shape[1]) if len(np.unique(self.data[:,i]).flatten())<10]
        included[nondiverse] = 0
        self.data = self.data[:,included]
        self.columns = self.columns[included]

        UVpattern = r'^UV\d+.*'
        Vpattern = r'^V\d+.*'
        Bpattern = r'^B\d+.*'
        YGpattern = r'^YG\d+.*'
        Rpattern = r'^R\d+.*'
        ImgBpattern = r'^ImgB\d+.*'
        Imagingpattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*'

        patterns = [UVpattern,Vpattern,Bpattern,YGpattern,Rpattern,ImgBpattern,Imagingpattern]
        self.patternmatches = np.ones(len(self.columns))*len(patterns)
        self.patternmatches = self.patternmatches.astype(int)

        for k,p in enumerate(patterns):
                matches = [i for i,c in enumerate(self.columns) if re.match(p,c)]
                self.patternmatches[matches] = k
        sort = np.argsort(self.patternmatches)
        self.patternmatches = self.patternmatches[sort]
        self.columns = self.columns[sort]
        self.data = self.data[:,sort]

        self.fcolors = np.array([self.colors[c] for c in self.patternmatches])
        self.flabels = np.array([self.ftypes[c] for c in self.patternmatches])
        
        self.filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types]
        self.patternmatches = self.patternmatches[self.filter]
        self.columns = self.columns[self.filter]
        self.data = self.data[:,self.filter]
        self.flabels = self.flabels[self.filter]
        self.fcolors = self.fcolors[self.filter]

        if norm:
                self.data = StandardScaler().fit_transform(self.data)


    def add_result(self, result):
        value = result['value']
        i =  result['i']
        self.medoids[list(result['medoids'].astype(int)),i] += 1
        self.memberships[:,i] = result['membership']
        self.feature_averages[:,i] = value
        non0 = np.any(self.feature_averages>0,axis=0)
        mean_value = np.mean(self.feature_averages[:,non0],axis=1).flatten()
        mdds = np.sum(self.medoids[:,non0],axis=1).flatten()
        self.result = {'ls': mean_value,'i': i,'medoids': mdds,'membership':result['membership']}

    def update_display(self):
        self.selected_feature_types = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]
        if hasattr(self, 'result'):
            filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types]
            self.output_layout.removeWidget(self.output_widget)
            self.output_widget = QWidget()
            self.output_layout = QVBoxLayout()
            self.output_widget.setLayout(self.output_layout)
            self.output_panel.setWidget(self.output_widget)

            mean_value = 1-self.NormalizeData(self.result['ls'])[filter]

            # Sort the results based on the dropdown selection
            sorting = True
            if "Sort by: Importance" in self.sort_dropdown.currentText():
                sort = np.argsort(-mean_value)
                sorting = False
            else:
                second = -mean_value
                if "Sort by: Type" in self.sort_dropdown.currentText():
                    first = self.flabels[filter]
                elif "Sort by: Centrality" in self.sort_dropdown.currentText():
                    first = -self.result['medoids'][filter]
                elif "Sort by: Cluster" in self.sort_dropdown.currentText() and self.finalcluster:
                    first = self.membership[filter]
                else:#If nothing else works (i.e. clustering not ready) then sort by Importance
                    sort = np.argsort(second)
                    sorting = False
            if sorting:
                sort = np.lexsort([second,first])
                sorting = False

            colors = self.fcolors[filter][sort]
            mean_value = mean_value[sort]
            medoids = self.result['medoids'][filter][sort]
            # topmeds = np.argsort(medoids)[::-1][:MEDS]
            topmeds = np.where(medoids>0)[0]
            texts = self.columns[filter][sort]
            labels = self.flabels[filter][sort]


            self.progress_bar.setValue(self.worker.progress)
            if self.finalcluster:
                membership = self.membership[filter][sort]
                # membership = self.membership.astype(int)
                memcolors = [self.clustercolors[m] for m in membership]

            for i in range(len(filter)):
                # Create a layout for each entry
                entry_layout = QHBoxLayout()
                text = texts[i]

                # Create and style the label for the colored text
                text_label = QLabel(text)
                if self.finalcluster:
                    if i in topmeds:
                        text_label.setStyleSheet(f"color: {colors[i]};font-weight: bold;border: 3px solid {memcolors[i]};text-decoration: underline")
                    else:
                        text_label.setStyleSheet(f"color: {colors[i]};border: 3px solid {memcolors[i]};")
                    entry_layout.addWidget(text_label)
                else:
                    if i in topmeds:
                        text_label.setStyleSheet(f"color: {colors[i]};font-weight: bold;text-decoration: underline")
                    else:
                        text_label.setStyleSheet(f"color: {colors[i]};")
                    entry_layout.addWidget(text_label)

                # Create and style the bar for the value
                bar = QFrame()
                bar.setStyleSheet(f"background-color: {colors[i]};")
                bar.setFixedHeight(10)
                bar.setFixedWidth(int(mean_value[i] * 300))  # Adjust multiplier for visual effect
                entry_layout.addWidget(bar)

                # Create a container widget for the entry layout
                entry_widget = QWidget()
                entry_widget.setLayout(entry_layout)
                # Add the entry widget to the output layout
                self.output_layout.addWidget(entry_widget)

            self.output_widget.adjustSize()
            QApplication.processEvents()

    def consensusclustering(self):
        memlabels = np.unique(self.memberships.flatten())
        D = np.zeros([self.memberships.shape[0],self.memberships.shape[0]])
        for m in memlabels:
            mem = (self.memberships == m)*1.
            D += mem @ mem.T
        np.fill_diagonal(D,0)
        # D /= self.memberships.shape[1]
        self.membership = np.array(la.find_partition(ig.Graph.Adjacency(D), la.ModularityVertexPartition).membership)
        self.finalcluster = True

    def finalize_results(self):
        self.output_widget.adjustSize()
        self.consensusclustering()
        self.update_display()
        self.result['Relative Importance'] = 1 - self.NormalizeData(self.result['ls'])
        self.result['Centrality'] = self.NormalizeData(self.result['medoids'])
        self.result['Membership'] = self.membership
        QMessageBox.information(self, "Information", "Processing complete!")
        self.update_timer.stop()  # Stop the update timer

    def save_output(self):
        if not self.result:
            QMessageBox.warning(self, "Warning", "There is no output to save.")
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "CSV Files (*.csv)", options=options)
        if filepath:
            try:
                with open(filepath, 'w', newline='') as csvfile:
                    fieldnames = ['feature','ri', 'ls','membership','centrality']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    result = self.result['ls']
                    impresult = self.result['Relative Importance']
                    columns = self.columns
                    memb = self.result['Membership']
                    centrality = self.result['Centrality']
                    for i in range(len(result)):
                        writer.writerow({'feature': columns[i], 'ri': impresult[i], 'ls': result[i],'membership':memb[i],'centrality': centrality[i]})
                QMessageBox.information(self, "Success", "Output successfully saved to CSV file.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save output to CSV file: {e}")

    def show_readme(self):
        readme_text = """
        This is the README for the Feature Importance Program.

        1. Enter the .fsc file path manually or click 'Browse' to select a file.
        2. Click 'Execute' to calculate the feature importance.
        3. Features will be shown from most to least important.
        4. The bar next to the feature name corresponds to relative feature importance.
        5. Use the boxes to toggle which types of features to display.
        6. De/selected boxes and executing allows you to test a subset of features.
        7. Use the 'File' menu to save the output as a CSV file.
        8. ri = (Relative) Importance, ls = Raw (Laplacian) Score, membership = Cluster, centrality = Representativeness
        """
        QMessageBox.information(self, "README", readme_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
