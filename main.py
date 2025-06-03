import sys
import os
import csv
import flowio
import flowkit as fk
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import pairwise_distances,adjusted_mutual_info_score
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

import re

import matplotlib
import matplotlib.colors as mcolors

import numpy as np; 
import traceback
import leidenalg as la
import igraph as ig



#Feature Design packages
import tifffile
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_otsu,gaussian
from skimage.morphology import binary_opening,binary_closing #,disk,local_minima,remove_small_objects
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max,canny
from scipy.ndimage import distance_transform_edt


# timer for evaluation
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QCheckBox, QPushButton, QProgressBar, QLabel, QFileDialog, QScrollArea, QFrame,
                             QAction, QMessageBox,QComboBox,QTabWidget,QSplitter, QFileSystemModel,QTreeView,QSlider,QMenu,QTextEdit,QSizePolicy,QDialog,QActionGroup)
from PyQt5.QtCore import QThread, pyqtSignal,  QTimer, Qt,QDir

from PyQt5.QtGui import QPixmap, QImage

VAL = False
#SIM = False
BOOT = 200
CLUSTERS = 10
MEDS = CLUSTERS
BOOTSIZE = 1000
THRESHOLD = 1e-2
KFRAC = 1./3
FOOTPRINT = generate_binary_structure(rank=2, connectivity=2)

# %matplotlib ipympl

matplotlib.use('Qt5Agg')
excludedcols = ['Saturated', 'Time', 'Sorted', 'Row', 'Column']
excludedcols += ['Protocol', 'EventLabel', 'Regions0', 'Regions1', 'Regions2',
       'Regions3', 'Gates', 'IndexSort', 'SaturatedChannels', 'PhaseOffset',
       'PlateLocationX', 'PlateLocationY', 'EventNumber0', 'EventNumber1',
       'DeltaTime0', 'DeltaTime1', 'DropId', 'SaturatedChannels1',
       'SaturatedChannels2', 'SpectralEventWidth', 'EventWidthInDrops',
       'SpectralUnmixingFlags', 'WaveformPresent']


class OperationHistory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_text_edit = QTextEdit()
        self.history_text_edit.setReadOnly(True)
        self.info_label = QLabel("No Image Loaded")
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Operation History:"))
        self.layout.addWidget(self.history_text_edit)
        self.layout.addWidget(self.info_label)
        self.setLayout(self.layout)

    def add_operation(self, operation_description):
        current_text = self.history_text_edit.toPlainText()
        new_text = f"{current_text}\n{operation_description}".strip()
        self.history_text_edit.setText(new_text)
        self.history_text_edit.verticalScrollBar().setValue(self.history_text_edit.verticalScrollBar().maximum())

    def update_info(self,info):
        self.info_label.setText(info)

# Path to the global CSV file containing feature names

class WorkerThread(QThread):
    progress_update = pyqtSignal(int)
    intermediate_result = pyqtSignal(dict)
    result_ready = pyqtSignal()

    def __init__(self, data):
        super().__init__()
        self.data = data
        
        N = self.data.shape[0]
        self.n=BOOTSIZE
        self.boots = BOOT
        if N<self.n:
            self.n = int(max([N/2,2]))
            self.boots = N
        self.k = max([int(self.n*KFRAC),1])
        self.mode = 'cosine'
        self.t = 1
        self.progress = 0
        self.early = 0

    def run(self):
        for i in range(self.boots):
            result = self.process_part(i)
            self.intermediate_result.emit(result)
            self.progress += 1
            if self.early:
                break
        self.result_ready.emit()

    def process_part(self, i):
        ls,medoids,medlabels = self.get_ulscore_parralel()
        return {"value": ls,"i": i,"medoids": medoids,"membership":medlabels}

    def getclust(self,mems):
        memlabels = np.unique(mems.flatten())
        D = np.zeros([mems.shape[0],mems.shape[0]])
        for m in memlabels:
            mem = (mems == m)*1.
            D += mem @ mem.T
        np.fill_diagonal(D,0)
        return np.array(la.find_partition(ig.Graph.Adjacency(D), la.ModularityVertexPartition).membership)
    
    def kmedoids(self,X):
        if CLUSTERS<=self.data.shape[1]/20:
            clusters = CLUSTERS
        else:
            clusters = int(self.data.shape[1]/20)
        model = KMedoids(n_clusters=clusters,method='pam').fit(X)
        medoids = model.medoid_indices_
        medlabels = model.labels_
        return medoids,medlabels
    
    def get_ulscore_parralel(self):
        n = self.n
        ones = np.ones((n,1))
        sample = np.random.choice(self.data.shape[0],n)
        Xsub = self.data[sample,:]
        Wsub = self.get_similaritymatrix(Xsub)
        Dsub = np.diagflat(np.sum(Wsub,axis=0))
        Lsub = Dsub - Wsub
        LSsub = np.zeros(Xsub.shape[1])
        for r in range(Xsub.shape[1]):#iterate over features
            fsubr = Xsub[:,r].reshape([-1,1])
            neighb_est = ((fsubr.T @ Dsub @ ones).item()/ (ones.T @ Dsub @ ones).item())*ones
            fsubr_est = (fsubr - neighb_est)#subtract nbh mean est of feature to centre feature vector
            d = (fsubr_est.T @ Dsub @ fsubr_est).item()
            num = (fsubr_est.T @ Lsub @ fsubr_est).item()
            if d > 0 and num>0:
                LSsub[r] = num/d
            elif num==0 and d>0:
                LSsub[r] = 0
            else:
                LSsub[r] = np.inf
        medoids,medlabels = self.kmedoids(Xsub.T)
        return LSsub,medoids,medlabels

    def get_similaritymatrix(self,X):

        t = self.t
        k = self.k
        n = X.shape[0]

        # compute pairwise distances
        D = self.getpwd(X,self.mode)
        Dtop = np.sort(D, axis=1)[:,k+1]
        G = D<=Dtop
        np.fill_diagonal(G,0)
        W = np.zeros([n,n])
        if self.mode=='heat':
            W[G>0] = np.exp(-D[G>0]**2/(2*t**2))
        else:#cosine is default
            W[G>0] = np.abs(1-D[G>0])
        return W
    
    def getpwd(self,X,mode='cosine'):
        if mode == 'heat':#heat kernel based pwd (euclidean)
            D = pairwise_distances(X)
        if mode == 'cosine':#cosine pwd
            D = pairwise_distances(X,metric='cosine')
        return D


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FlowFI: Flow cytometry Feature Importance")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        # self.tabs.resize(300,200)

        self.tabs.addTab(self.tab1,"Analysis")
        self.tabs.addTab(self.tab2,"Design")
        self.tab1.layout = QVBoxLayout(self.tab1)
        self.tab2.layout = QVBoxLayout(self.tab2)
        self.layout.addWidget(self.tabs)

        # TAB-1 LAYOUT: ANALYSIS


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
        
        centrality_checkbox = QCheckBox('CEN ONLY')
        centrality_checkbox.setChecked(False)
        centrality_checkbox.stateChanged.connect(self.update_display)
        self.centrality_checkbox = centrality_checkbox
        self.checkbox_layout.addWidget(self.centrality_checkbox)

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
        self.sort_dropdown.addItem("Sort by: Change from Previous Importance (contrast scores against previous run)")
        # self.sort_dropdown.setItemData(4, False, Qt.ItemIsEnabled)

        self.sort_dropdown.currentIndexChanged.connect(self.attempt_sort)
        
        self.tab1.layout.addLayout(self.checkbox_layout)
        self.tab1.layout.addLayout(self.input_layout)
        self.tab1.layout.addWidget(self.execute_button)
        self.tab1.layout.addWidget(self.progress_bar)
        self.tab1.layout.addWidget(self.sort_dropdown)
        self.tab1.layout.addWidget(QLabel("Feature/Importance:"))
        self.tab1.layout.addWidget(self.output_panel)
        self.finalcluster = False

        self.tab1.setLayout(self.tab1.layout)
        self.central_widget.setLayout(self.layout)

        #TAB-2 DESIGN LAYOUT



        self.operation_history = []
        self.operations_performed = 0
        self.current_channel = None
        self.current_image_array = None
        self.processed_image = None
        self.agg_operation = None

        # Root directory input
        root_path_layout = QHBoxLayout()
        self.root_path_input = QLineEdit(QDir.homePath())
        self.root_path_input.returnPressed.connect(self.set_tree_root)

        self.change_root_button = QPushButton("Change Root")
        self.change_root_button.clicked.connect(self.browse_for_root)

        root_path_layout.addWidget(self.root_path_input)
        root_path_layout.addWidget(self.change_root_button)
        self.tab2.layout.addLayout(root_path_layout)

        # File system tree
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.homePath())
        self.model.setNameFilters(["*.tiff", "*.tif"])
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.homePath()))

        # Image display (left)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_label.setFixedHeight(100) 

        # Processed image display (right)
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFrameShape(QFrame.StyledPanel)
        self.processed_image_label.setText("Processed Image") # Initial text
        self.processed_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.processed_image_label.setFixedHeight(100)   
    
        # Channel slider
        self.channel_label = QLabel("Channel: ")
        self.channel_slider = QSlider(Qt.Vertical)
        self.channel_slider.valueChanged.connect(self.update_displayed_channel)
        self.channel_slider.setEnabled(False) # Disable initially

        # Create a vertical layout for each side of the split
        left_image_panel = QWidget()
        left_layout = QVBoxLayout(left_image_panel)
        left_layout.addWidget(self.image_label)
        
        right_image_panel = QWidget()
        right_layout = QVBoxLayout(right_image_panel)
        right_layout.addWidget(self.processed_image_label)

        # Create a vertical layout for each side of the split
        channel_panel = QWidget()
        channel_layout = QVBoxLayout(channel_panel)
        channel_layout.addWidget(self.channel_label)
        channel_layout.addWidget(self.channel_slider)


        # Create a horizontal splitter for the image panels
        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_splitter.addWidget(left_image_panel)
        self.image_splitter.addWidget(right_image_panel)
        self.image_splitter.addWidget(channel_panel)
        self.image_splitter.setSizes([180,180,20]) # Adjust initial sizes

        self.terminal = OperationHistory()


        self.reset_operations_button = QPushButton("Reset")
        self.reset_operations_button.clicked.connect(self.reset_operations)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.image_splitter)
        right_layout.addWidget(self.reset_operations_button)
        right_layout.addWidget(self.terminal)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 400])

        self.tab2.layout.addWidget(splitter) # Add the splitter to the tab's layout

       # Connect the tree view's double-click signal to the image loading function
        self.tree.doubleClicked.connect(self.load_tiff_image)

        # Menu bar
        self.create_menus()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        # self.update_timer.timeout.connect(self.update_progress)
    
    #TAB-2 Functions
    
    def browse_for_root(self):
        directory = QFileDialog.getExistingDirectory(self, "Select New Root Directory",
                                                   self.root_path_input.text(),
                                                   QFileDialog.ShowDirsOnly)
        if directory:
            self.root_path_input.setText(directory)
            self.set_tree_root()

    def set_tree_root(self):
        root_path = self.root_path_input.text()
        if QDir(root_path).exists():
            self.model.setRootPath(root_path)
            self.tree.setRootIndex(self.model.index(root_path))
        else:
            print(f"Error: Root path '{root_path}' does not exist.")

    def load_tiff_image(self, index):
        self.processed_image = None
        self.operations_performed = 0
        file_path = self.model.filePath(index)
        self.tree.scrollTo(index)
        self.tree.setCurrentIndex(index)
        if file_path.lower().endswith(('.tiff', '.tif')):
            try:
                tif_image = tifffile.imread(file_path)
                self.current_image_array = np.array(tif_image)

                if self.current_image_array.ndim >= 3:
                    # Assuming channels are the first or last dimension
                    # You might need to adjust this based on your TIFF structure
                    if self.current_image_array.shape[0] > 1:
                        self.num_channels = self.current_image_array.shape[0]
                    elif self.current_image_array.shape[-1] > 1:
                        self.num_channels = self.current_image_array.shape[-1]
                    else:
                        self.num_channels = 1
                        self.current_image_array = np.expand_dims(self.current_image_array, axis=0) # Add a channel dimension

                    self.channel_slider.setMinimum(0)
                    self.channel_slider.setMaximum(self.num_channels - 1)
                    self.channel_slider.setEnabled(True)
                    if self.current_channel is None:
                        self.channel_slider.setValue(0)
                        self.update_displayed_channel(0) # Display the first channel
                    else:
                        if self.num_channels>self.current_channel>=0:#if channel does not exist for new image
                            self.channel_slider.setValue(self.current_channel)
                            self.update_displayed_channel(self.current_channel)
                        else:
                            self.channel_slider.setValue(0)
                            self.update_displayed_channel(0) # Display the first channel
                    self.terminal.add_operation('Image Set: ' + str(file_path).split('/')[-1])
                    self.terminal.update_info(f"Array Info: Shape={self.current_image_array.shape}, Dtype={self.current_image_array.dtype}")

                elif self.current_image_array.ndim == 2:
                    self.current_image_array = np.expand_dims(self.current_image_array, axis=0) # Treat as single channel
                    self.num_channels = 1
                    self.channel_slider.setEnabled(False)
                    self.update_displayed_channel(0)
                    self.terminal.add_operation('Image Set: ' + str(file_path).split('/')[-1])
                    self.terminal.update_info(f"Array Info: Shape={self.current_image_array.shape}, Dtype={self.current_image_array.dtype}")
                    
                else:
                    self.terminal.add_operation("Not a suitable image format for channel viewing.")
                    self.channel_slider.setEnabled(False)
                    self.terminal.update_info("No Image Loaded")
                    self.current_image_array = None
                    self.num_channels = 0
                if self.current_image_array is not None:
                    self.image_menu.setEnabled(True)

            except ImportError:
                self.image_label.setText("Error: Required library not found.")
            except Exception as e:
                self.image_label.setText(f"Error loading TIFF file: {e}")
                self.channel_slider.setEnabled(False)
                self.current_image_array = None
                self.num_channels = 0
        else:
            self.image_label.clear()
            self.channel_slider.setEnabled(False)
            self.current_image_array = None
            self.num_channels = 0

    def update_displayed_channel(self, channel_index):
        self.operations_performed = 0
        if self.current_image_array is not None and 0 <= channel_index < self.num_channels:
            self.terminal.add_operation(f"Channel Set to:  {channel_index+1}")
            self.channel_label.setText(f"Channel: {channel_index + 1}/{self.num_channels}")
            self.current_channel = channel_index
            self.processed_image = self.current_image_array[self.current_channel]

            # Normalize and convert to 8-bit grayscale for display
            normalized_array = self.norm(self.current_image_array[channel_index])
            height, width = normalized_array.shape
            self.current_q_image = QImage(normalized_array.data, width, height, width, QImage.Format_Grayscale8)

            self.update_left_image_label()
            self.process_image()
    
    def update_left_image_label(self):
        if self.current_q_image is not None:
            pixmap = QPixmap.fromImage(self.current_q_image)
            scaled_pixmap = pixmap.scaled(
                self.current_q_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.clear()
    
    def reset_operations(self):
        self.operation_history = []
        self.operations_performed = 0
        self.processed_image = self.current_image_array[self.current_channel]
        self.terminal.add_operation('Reset Operations')
        self.process_image()
    
    def norm(self,array,eightbit=True):
        array -= np.min(array)
        array /= np.max(array)
        array *= 255
        if eightbit:
            array = np.round(array).astype('uint8')
        return array
    
    def enable_aggregation(self,action):
        if action == self.count_action:
            self.enable_count()
        elif action == self.mean_action:
            self.enable_mean()
        elif action == self.area_action:
            self.enable_area()
    
    def enable_area(self):
        self.agg_operation = 'area'
        self.terminal.add_operation('Feature set to: Area')
        self.process_image()
    
    def enable_mean(self):
        self.agg_operation = 'mean'
        self.terminal.add_operation('Feature set to: Mean')
        self.process_image()
    
    def enable_count(self):
        self.agg_operation = 'count'
        self.terminal.add_operation('Feature set to: Count')
        self.process_image()
    
    def do_aggregation(self):
        uniq = np.unique(self.processed_image)
        luniq = len(uniq)
        if luniq>1:
            if self.agg_operation == 'area':
                if 0 in uniq:
                    area = self.get_area(self.processed_image)
                    self.terminal.add_operation(f"Area is: {area}")
            if self.agg_operation == 'mean':
                mean = self.get_mean(self.processed_image)
                self.terminal.add_operation(f"Mean is: {mean}")
            if self.agg_operation == 'count':
                count = self.get_count(self.processed_image)
                self.terminal.add_operation(f"Count is: {count}")

    def do_aggregation_silent(self,image):
        uniq = np.unique(image)
        luniq = len(uniq)
        score = np.nan
        if luniq>1:
            if self.agg_operation == 'area':
                if 0 in uniq:
                    score = self.get_area(image)
            elif self.agg_operation == 'mean':
                score = self.get_mean(image)
            elif self.agg_operation == 'count':
                score = self.get_count(image)
            else:#default to count
                score = self.get_count(image)
        else:
            score = np.nan
        if np.isnan(score):
            return 0
        else:
            return score             



    def process_image(self):
        # Placeholder for your image processing operation
        self.perform_operations()
        height, width = self.processed_image.shape
        pimage = self.norm(self.processed_image).data
        self.processed_q_image = QImage(pimage, width, height, width, QImage.Format_Grayscale8)
        self.update_right_image_label()

    def update_right_image_label(self):
        if self.processed_q_image is not None:
            pixmap = QPixmap.fromImage(self.processed_q_image)
            scaled_pixmap = pixmap.scaled(
                self.processed_q_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.processed_image_label.setPixmap(scaled_pixmap)
        else:
            self.processed_image_label.clear()
    
    def perform_operations(self):
        nops = len(self.operation_history)
        for i in range(self.operations_performed,nops):
            self.do_operation(i)
        self.operations_performed = nops
        if self.agg_operation is not None:
            self.do_aggregation()
    
    def do_operation(self,index):
        operation = self.operation_history[index]
        # self.terminal.add_operation(str(operation[0]))
        if operation[0]=='gauss':
            self.processed_image = self.gaussblur(self.processed_image, float(operation[1]))  # Call self.gauss
            self.terminal.add_operation(f'Gaussian Blur: {np.round(float(operation[1]),2)} Channel: {self.current_channel+1}')
        if operation[0]=='mask':
            self.processed_image = self.get_mask(self.processed_image.astype(float)).astype(float)
            self.terminal.add_operation(f'Mask Channel: {self.current_channel+1}')
        if operation[0]=='label':
            self.processed_image = self.get_label(self.processed_image.astype(int)).astype(float)
            self.terminal.add_operation(f'Label Channel: {self.current_channel+1}')
        if operation[0]=='segment':
            self.processed_image = self.get_segment(self.processed_image.astype(float)).astype(float)
            self.terminal.add_operation(f'Segment Channel: {self.current_channel+1}')
    
    def do_operation_silent(self,index,image):
        operation = self.operation_history[index]

        if operation[0]=='gauss':
            return self.gaussblur(image, float(operation[1]))  # Call self.gauss
        if operation[0]=='mask':
            return self.get_mask(image.astype(float)).astype(float)
        if operation[0]=='label':
            return self.get_label(image.astype(int)).astype(float)
        if operation[0]=='segment':
            return self.get_segment(image.astype(float)).astype(float)

            
    # def resizeEvent(self, event):
    #     if self.current_image_array is not None:
    #         self.update_left_image_label()
    #         # current_channel_index = self.channel_slider.value()
    #         # self.processed_image = self.current_image_array[current_channel_index]
    #         self.process_image()
    #     super().resizeEvent(event)
    
    def open_gauss(self):
        dialog = GaussDialog(self)  # Pass self as parent
        if dialog.exec_() == QDialog.Accepted:
            sigma = dialog.get_sigma()
            self.operation_history += [['gauss',sigma]]
            # self.terminal.add_operation(f'Gaussian Blur: {np.round(sigma,2)} Channel: {self.current_channel+1}')
            self.process_image()
        else:
            print("Dialog cancelled.")

    def do_mask(self):
        self.operation_history += [['mask']]
        # self.terminal.add_operation(f'Mask Channel: {self.current_channel+1}')
        self.process_image()

    def do_segment(self):
        self.operation_history += [['segment']]
        # self.terminal.add_operation(f'Segment Channel: {self.current_channel+1}')
        self.process_image()
    
    def do_label(self):
        self.operation_history += [['label']]
        # self.terminal.add_operation(f'Label Channel: {self.current_channel+1}')
        self.process_image()
    
    def do_export_csv(self):
        self.do_process_images(mode='csv')


    def do_export_fcs(self):
        self.do_process_images(mode='fcs')
    
    
    def do_process_images(self,mode='image'):
        if mode == False:#Edge case
            mode = 'image'
        if self.current_image_array is None:
            QMessageBox.warning(self, "Warning", "No image is currently displayed. Please open or display an image first.")
            return
        if self.current_channel is None:
            QMessageBox.warning(self, "Warning", "Could not determine the number of channels in the currently displayed image.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if not folder_path:
            return  # User cancelled the folder selection
        

        processed_count = 0
        if mode == 'image':
            pparent = os.path.dirname(folder_path)
            ppath = os.path.join(pparent,'processed')
            os.makedirs(ppath,exist_ok=True)
        elif mode=='csv' or mode=='fcs':
            vals = []
            if mode == 'csv':
                vfile = os.path.join(folder_path,'new_parameter.csv')
            elif mode=='fcs':
                try:
                    old_fcsfile = old_fcsfile = self.get_fcs_files(folder_path)[0]
                    new_fcsfile = str(old_fcsfile)[:-4] + '_new.fcs'
                except Exception as e:
                    QMessageBox.critical(self, "Error", "No suitable fcs file found in directory")

        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                filename = os.path.join(subdir,file)
                if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
                    filepath = filename
                    try:
                        # Use Pillow to open the TIFF image and get its number of bands (channels)
                        img = np.array(tifffile.imread(filepath))
                        tiff_channels = img.shape[0]
                        img = img[self.current_channel,:,:]

                        if tiff_channels > self.current_channel:
                            # Add your actual image processing logic here for multichannel grayscale images
                            # For example:
                            # processed_image = self.your_processing_function(filepath)
                            # Save the processed image
                            processed_image = self.process_image_export(img)
                            if mode == 'image':
                                print(processed_image.shape)
                                tifffile.imwrite(os.path.join(ppath, f"processed_{file}"),np.expand_dims(processed_image.astype('float32'), axis=0))
                            elif mode == 'csv' or mode == 'fcs':
                                vals.append(self.do_aggregation_silent(processed_image))
                            processed_count += 1
                        else:
                            print(f"Skipping: {filename} (insufficient number of channels))")

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error processing {filename}: {e}")
        if mode=='image':
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} TIFF files into: {ppath}")
        elif mode == 'csv':
            self.param_to_csv(vals,vfile)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {vfile}")
        elif mode == 'fcs':
            self.param_to_fcs(vals,old_fcsfile,new_fcsfile)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {new_fcsfile}")

    def process_image_export(self,image):
        for i,op in enumerate(self.operation_history):
            image = self.do_operation_silent(i,image)
        return image
    
    def param_to_csv(self,vals,vfile):
        vals = np.array(vals).reshape(-1,1)
        with open (vfile,'w') as f:
            wtr = csv.writer(f)
            wtr.writerows(vals)
    
    def param_to_fcs(self,vals,ofcs,nfcs):
        vals = 10^np.array(vals).reshape(-1,1)
        fcs,metadata = self.load_fcs(ofcs)
        self.add_param(fcs,nfcs,metadata,vals)
    
    def load_fcs(self,fcsfile):
        fcdata = flowio.FlowData(fcsfile)
        fcsample = fk.Sample(fcsfile)
        metadata = fcsample.metadata
        return fcdata,metadata
    
    def add_param(self,fcdata,nfcs,metadata,vals,pname='new_param'):
        numc = fcdata.channel_count
        channels = [fcdata.channels[k]['PnN'] for k in fcdata.channels.keys()]
        events = np.reshape(fcdata.events,(-1,numc))
        channels.append(pname)
        events = np.hstack([events,vals])
        print(events.shape)
        events = events.flatten()
        flowio.create_fcs(open(nfcs,'wb'),events,channels,opt_channel_names=channels,metadata_dict=metadata)



    def get_fcs_files(self,directory):
        """
        Returns a list of all files in the given directory that have the suffix .fcs.

        Args:
            directory (str): The path to the directory to search.

        Returns:
            list: A list of the full paths to the .fcs files found.
                Returns an empty list if no .fcs files are found or if the
                directory does not exist.
        """
        fcs_files = []
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.lower().endswith(".fcs"):
                    full_path = os.path.join(directory, filename)
                    if os.path.isfile(full_path):  # Ensure it's a file, not a subdirectory
                        fcs_files.append(full_path)
        return fcs_files

    
    # Image functions

    def get_peaks(self,image,mind=10):
        coordinates = peak_local_max(
            image, 
            min_distance=mind,  # Controls separation between peaks
            threshold_abs=0.01,# Ignores low-intensity peaks
            exclude_border=False
        )
        peakimage = np.zeros(image.shape)
        peakimage[coordinates[:,0],coordinates[:,1]]=image[coordinates[:,0],coordinates[:,1]]
        
        return peakimage

    def get_segment(self,image):
        labmask = label(image)
        if np.max(labmask)!=1:
            segmented = labmask
        else:
            edges = canny(labmask.astype('float'),sigma=1)
            distance = distance_transform_edt(edges)  # Compute distance from edges
            markers = label(self.get_peaks(distance,20)*labmask) 
            segmented = watershed(-distance,markers=markers,mask=labmask)
            if np.sum(segmented>0)==0:
                segmented = labmask
        return segmented

    def get_mask(self,image,clopen=True):
        mask = (image>=threshold_otsu(image))
        if not clopen:
            return mask
        else:
            return binary_closing(binary_opening(mask,footprint=FOOTPRINT),footprint=FOOTPRINT)
    
    def get_area(self,image):
        return np.sum(image!=0)
    
    def get_mean(self,image):
        return np.mean(image!=0)
    
    def get_count(self,image):
        uniq = np.unique(image)
        luniq = len(uniq)
        if 0 in uniq:
            luniq -= 1
        return luniq
    
    def get_label(self,image):
        return label(image)
    
    def gaussblur(self,image,sigma=2):
        return gaussian(image,sigma,mode='wrap')
    
    # MENU FUNCTIONS

    def create_menus(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu('File')

        save_action = QAction('Save Output as CSV', self)
        save_action.triggered.connect(self.save_output)
        file_menu.addAction(save_action)

        load_action = QAction('Load Output CSV for Comparison', self)
        load_action.triggered.connect(self.compare_output)
        file_menu.addAction(load_action)

        # Help menu
        help_menu = menu_bar.addMenu('Help')

        readme_action = QAction('README', self)
        readme_action.triggered.connect(self.show_readme)
        help_menu.addAction(readme_action)

        # Image Menu
        self.image_menu = menu_bar.addMenu('Image')

        # Image Submenus

        manipulation_submenu = QMenu('&Image Operations', self)
        filters_submenu = QMenu('&Filter', self)
        gauss = QAction('&Gaussian Filter',self)
        gauss.triggered.connect(self.open_gauss)
        filters_submenu.addAction(gauss)

        segmentation_submenu = QMenu('&Segmentation', self)
        mask = QAction('&Mask Otsu',self)
        mask.triggered.connect(self.do_mask)
        mlabel = QAction('&Label Image',self)
        mlabel.triggered.connect(self.do_label)  
        segment = QAction('&Segment',self)
        segment.triggered.connect(self.do_segment)      
        segmentation_submenu.addAction(mask)
        segmentation_submenu.addAction(segment)
        segmentation_submenu.addAction(mlabel)
        process_images_action = QAction('Process Images',self)
        process_images_action.triggered.connect(self.do_process_images)


        manipulation_submenu.addMenu(filters_submenu)
        manipulation_submenu.addMenu(segmentation_submenu)
        manipulation_submenu.addAction(process_images_action)


        feature_submenu = QMenu('&Image to Parameter', self)

        # Create actions for the mutually exclusive options
        self.count_action = QAction("Count (unique)", self, checkable=True)
        self.mean_action = QAction("Mean (non-zero)", self, checkable=True)
        self.area_action = QAction("Area (non-zero)", self, checkable=True)
    
        # Create an action group to make them mutually exclusive
        aggregation_group = QActionGroup(self)
        aggregation_group.triggered.connect(self.enable_aggregation)
        aggregation_group.addAction(self.count_action)
        aggregation_group.addAction(self.mean_action)
        aggregation_group.addAction(self.area_action)



        aggregation_submenu = QMenu('&Aggregate', self)
        geometry_submenu = QMenu('&Geometry', self)
        export_to_fcs = QAction('Export to FCS',self)
        export_to_fcs.triggered.connect(self.do_export_fcs)
        export_to_csv = QAction('Export to CSV',self)
        export_to_csv.triggered.connect(self.do_export_csv)
        feature_submenu.addMenu(aggregation_submenu)
        feature_submenu.addMenu(geometry_submenu)
        feature_submenu.addActions([export_to_fcs,export_to_csv])

        # Upcoming Features
        solidity_action = QAction('&Solidity',self)
        solidity_action.setEnabled(False)
        geometry_submenu.addAction(solidity_action)
        coloc_action = QAction('&Colocalisation',self)
        coloc_action.setEnabled(False)
        geometry_submenu.addAction(coloc_action)
        peaks_action = QAction('&Find Peaks',self)
        peaks_action.setEnabled(False)
        geometry_submenu.addAction(peaks_action)  
        breg_action = QAction('&Bregnan Denoising',self)
        breg_action.setEnabled(False)
        filters_submenu.addAction(breg_action)        


        self.image_menu.addMenu(manipulation_submenu)
        self.image_menu.addMenu(feature_submenu)

        aggregation_submenu.addAction(self.count_action)
        aggregation_submenu.addAction(self.mean_action)
        geometry_submenu.addAction(self.area_action)

        self.image_menu.setEnabled(False)



        # readme_action = QAction('README', self)

    # TAB-1 FUNCTIONS
    

    def compare_output(self):
        if self.finalcluster:
            self.load_output()
        else:
            QMessageBox.information(self, "Error", "No complete results to compare to yet.")
    
    def attempt_sort(self,index):
        if index == 4:
            if not hasattr(self,"loaded_result"):
                self.compare_output()
                if self.finalcluster==False:
                    self.sort_dropdown.setCurrentIndex(0)
                    self.update_display()
            else:
                self.update_display()
        else:
            self.update_display()
    

    def load_output(self,index=0):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Output CSV", "", "CSV Files (*.csv)", options=options)
        if filepath:
            try:
                loaded_result = {}
                with open(filepath, 'r') as csvfile:
                    loaded_result = {}
                    reader = csv.DictReader(csvfile)
                    fieldnames = reader.fieldnames
                    for f in fieldnames:
                        loaded_result[f] = []
                    for line in reader:
                        for f in line.keys():
                            loaded_result[f] += [line[f]]
                # print(fieldnames)
                for f in fieldnames:
                    if f!='feature':
                        loaded_result[f] = np.array(loaded_result[f]).astype('float')
                    else:
                        loaded_result[f] = np.array(loaded_result[f])
                self.loaded_result = loaded_result
                self.update_display()
                # QMessageBox.information(self, "Success", "Output successfully loaded from CSV file.")
            except Exception as e:
                self.sort_dropdown.setCurrentIndex(index)
                self.update_display()
                QMessageBox.critical(self, "Error", f"Failed to load output from CSV file: {e}")
    
    

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
        
        self.execute_button.setEnabled(False)
        self.start_time = time.time()
        self.output_layout.removeWidget(self.output_widget)
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)


        self.progress_bar.setValue(0)

        self.worker = WorkerThread(self.data)
        self.boots = self.worker.boots
        self.feature_averages = np.zeros((self.data.shape[1],self.boots))
        self.calculated = np.zeros((self.boots))
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
            # if SIM:
            #     n_features = self.data.shape[1]
            #     ninf = np.max([int(n_features*.01),1])
            #     nred = np.max([int(n_features*.01),1])
            #     cshape = int(self.data.shape[0]*.6)
            #     self.data,_ = make_classification(self.data.shape[0], n_features,n_repeated=0,weights=[.1],n_informative=ninf,n_redundant=nred,n_clusters_per_class=1,shuffle=False)
            #     ncontam = n_features - ninf - nred
            #     contam,_ = make_classification(cshape, ncontam,n_repeated=0,weights=[.5],n_informative=ncontam,n_redundant=0,n_clusters_per_class=1,shuffle=False)
            #     self.data[:cshape,-ncontam:] += contam
            #     self.meaningful = np.zeros(self.data.shape[1])
            #     self.meaningful[:(ninf+nred)] = 1

            
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
    
    def splittest(self,data,th=THRESHOLD):
        shape = data.shape[1]
        inds = np.arange(shape)
        np.random.shuffle(inds)
        data = data[:,inds]
        splitat = int(shape/2)
        inds1 = inds[:splitat]
        inds2 = inds[splitat:]
        data1 = np.mean(data[:,inds1],axis=1)
        data2 = np.mean(data[:,inds2],axis=1)
        kt = kendalltau(data1,data2)
        kt = kt.statistic
        if 1-kt<=th:
            return True,inds1,inds2
        else:
            return False,inds1,inds2
    
    def consensusclustering_test(self,inds1,inds2,th=THRESHOLD):
        mems = self.memberships[:,self.calculated>0]
        mems1 = mems[:,inds1]
        mems2 = mems[:,inds2]
        membership1 = self.worker.getclust(mems1)
        membership2 = self.worker.getclust(mems2)
        ami = adjusted_mutual_info_score(membership1,membership2)
        return ami>th

    def add_result(self, result):
        value = result['value']
        i =  result['i']
        self.medoids[list(result['medoids'].astype(int)),i] += 1
        self.memberships[:,i] = result['membership']
        self.feature_averages[:,i] = value
        self.calculated[i] = 1
        non0 = self.calculated>0
        imp_calculated = self.feature_averages[:,non0]
        mean_value = np.mean(imp_calculated,axis=1).flatten()
        mdds = np.sum(self.medoids[:,non0],axis=1).flatten()
        self.result = {'ls': mean_value,'i': i,'medoids': mdds,'membership':result['membership']}
        if np.sum(self.calculated)>10:
            isconv,inds1,inds2 = self.splittest(imp_calculated)
            if isconv:
                # print('The importance method converged at or before ',np.sum(self.calculated),' iterations')
                # if SIM:
                #     mean_value = 1 - self.NormalizeData(self.result['ls'])
                #     nmean = np.sum(self.meaningful)
                #     found = np.argsort(-mean_value)<nmean
                #     found = np.dot(found,self.meaningful)
                #     print('Acccuracy',np.round(100*found/nmean,2))
                isclust = self.consensusclustering_test(inds1,inds2)
                if isclust:
                    # print('The clustering also converged at or before ',np.sum(self.calculated),' iterations')
                    self.end_time = time.time()
                    self.total_time = np.round(self.end_time - self.start_time,2)
                    # print('Processing time was',self.total_time)
                    self.worker.early = 1

    def color_name_to_rgba(self,color_name):
        try:
            rgba = mcolors.to_rgba(color_name)
            return rgba
        except ValueError:
            return (0,0,0,0)

    def update_display(self):
        self.selected_feature_types = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]

        if hasattr(self, 'result'):
            if self.centrality_checkbox.isChecked():
                central_features = [i for i,m in enumerate(self.result['medoids']) if m>0]
                filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types and i in central_features]
            else:
                filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types]
            self.output_layout.removeWidget(self.output_widget)
            self.output_widget = QWidget()
            self.output_layout = QVBoxLayout()
            self.output_widget.setLayout(self.output_layout)
            self.output_panel.setWidget(self.output_widget)

            mean_value = 1-self.NormalizeData(self.result['ls'])[filter]
            
            loaded_final = self.finalcluster and hasattr(self,"loaded_result")
            if loaded_final:
                ffeatures = self.columns[filter]
                loaded_ffeatures = self.loaded_result['feature']

                loaded_orderedimp = np.zeros(len(ffeatures))
                orderedimp = np.zeros(len(ffeatures))
                for i in range(len(ffeatures)):
                    if ffeatures[i] in loaded_ffeatures:
                        ind = int(np.where(ffeatures[i]==loaded_ffeatures)[0][0])
                        loaded_orderedimp[i] = self.loaded_result['ls'][ind]
                        orderedimp[i] = self.result['ls'][filter][i]
                    else:
                        orderedimp[i] = -1
                        loaded_orderedimp[i] = -1
                    
                orderedimp[orderedimp>=0] = orderedimp[orderedimp>=0].argsort().argsort()
                loaded_orderedimp[loaded_orderedimp>=0] = loaded_orderedimp[loaded_orderedimp>=0].argsort().argsort()
                rankdiffs = np.zeros(len(ffeatures))
                rankdiffs[orderedimp>=0] = orderedimp[orderedimp>=0]-loaded_orderedimp[loaded_orderedimp>=0]
                rankdiffs[orderedimp==-1] = np.nan
                self.result['Comparison'] = -rankdiffs

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
                elif "Sort by: Change" in self.sort_dropdown.currentText() and loaded_final:
                    first = rankdiffs       
                else:#If nothing else works (i.e. clustering not ready) then sort by Importance
                    sort = np.argsort(second)
                    sorting = False
            if sorting:
                sort = np.lexsort([second,first])
                sorting = False

            colors = self.fcolors[filter][sort]
            mean_value = mean_value[sort]
            medoids = self.result['medoids'][filter][sort]
            topmeds = np.where(medoids>0)[0]
            texts = self.columns[filter][sort]
            labels = self.flabels[filter][sort]

            if loaded_final:
                rankdiffs = rankdiffs[sort]

            if self.worker.early:
                self.worker.progress = self.boots    
            prog = int(100*self.worker.progress/self.boots)
            self.progress_bar.setValue(prog)
            if self.finalcluster:
                membership = self.membership[filter][sort]
                # membership = self.membership.astype(int)
                memcolors = [self.clustercolors[m] for m in membership]

            for i in range(len(filter)):
                # Create a layout for each entry
                entry_layout = QHBoxLayout()
                text = texts[i]
                if loaded_final:
                    if -rankdiffs[i]>0:
                        text += ' (+' + str(int(-rankdiffs[i])) + ')'
                    elif -rankdiffs[i]<=0:
                        text += ' (' + str(int(-rankdiffs[i])) + ')'
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

    def show_processing_time(self):
        text = "Processing time: " + str(self.total_time) + 's'
        QMessageBox.information(self, "Processing Time", text)

    def consensusclustering_final(self):
        self.membership = self.worker.getclust(self.memberships)
        self.finalcluster = True
        if EVAL == True:
            self.end_time = time.time()
            self.total_time = np.round(self.end_time - self.start_time,2)
            self.show_processing_time()
        self.execute_button.setEnabled(True)

    def finalize_results(self):
        if self.worker.early:
            self.memberships = self.memberships[:,self.calculated>0]
            self.feature_averages = self.feature_averages[:,self.calculated>0]
            self.medoids = self.medoids[:,self.calculated>0]
        self.output_widget.adjustSize()
        self.consensusclustering_final()
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
                    if not hasattr(self,"loaded_result"):
                        fieldnames = ['feature','ri', 'ls','membership','centrality']
                    else:
                        fieldnames = ['feature','ri', 'ls','membership','centrality',"comparison"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    result = self.result['ls']
                    impresult = self.result['Relative Importance']
                    columns = self.columns
                    memb = self.result['Membership']
                    centrality = self.result['Centrality']
                    if not hasattr(self,"loaded_result"):
                        for i in range(len(result)):
                            writer.writerow({'feature': columns[i], 'ri': impresult[i], 'ls': result[i],'membership':memb[i],'centrality': centrality[i]})
                    else:
                        comparison = self.result['Comparison']
                        for i in range(len(result)):
                            writer.writerow({'feature': columns[i], 'ri': impresult[i], 'ls': result[i],'membership':memb[i],'centrality': centrality[i],'comparison': comparison[i]})
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
        5. Use the boxes to toggle which types of features to display./mnt/data/Dropbox/Work/Turing/240305_MKBM_fix_P13_images_20240308_1513_03/00000000/240305_MKBM_fix_P13_00000327.tiff
        6. De/selected boxes and executing allows you to test a subset of features.
        7. Use the 'File' menu to save the output as a CSV file.
        8. ri = (Relative) Importance, ls = Raw (Laplacian) Score, membership = Cluster, centrality = Representativeness
        """
        QMessageBox.information(self, "README", readme_text)

class GaussDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Blur Sigma")  # More specific title
        self.sigma = 0.0

        layout = QVBoxLayout(self)

        # Label and Line Edit for Sigma
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        self.sigma_edit = QLineEdit("2.0")
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.sigma_edit)
        layout.addLayout(sigma_layout)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)  # Connect Cancel

        self.setLayout(layout)

    def get_sigma(self):
        return self.sigma

    def accept(self):
        try:
            self.sigma = float(self.sigma_edit.text())
            if self.sigma <= 0:
                QMessageBox.warning(self, "Invalid Input", "Sigma must be greater than zero.", QMessageBox.Ok)
                return  # Stay in dialog if input is invalid
            super().accept()  # Close dialog and set result to Accepted
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for sigma.", QMessageBox.Ok)
            return  # Stay in dialog if input is not a number
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
