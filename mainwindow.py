
from ui_mainwindow import Ui_MainWindow
from enum import Enum
import sys
import time

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

from MACRO import *
from TextureAnalysis import *
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QApplication, QFileDialog



from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QApplication, QGraphicsPixmapItem, QListWidgetItem
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QImage, QPixmap
import nibabel as nib


        

class mainwindow(QMainWindow):
    
    start_singleImage_texture_analysis_sig = pyqtSignal(np.ndarray, np.ndarray, dict) # image, mask, settings
    start_dataset_texture_analysis_sig = pyqtSignal(str, dict) # dataset directory, settings

    def __init__(self):
        super(mainwindow, self).__init__()
        self.ui = Ui_MainWindow()
        
        self.ui.setupUi(self)
        self.uiInitialize()
        self.multiBtnSignalAvoider = 0
        
        self.img_graphicsview = GridGraphicsView(5,3)
        self.ui.graphicsViewGridLayout.addWidget(self.img_graphicsview, 2, 0, 1, 1)
        
        self.texture_analyser = TextureAnalysis()
        self.calculation_thread = QThread()
        # Move calculator to the thread(
        self.texture_analyser.moveToThread(self.calculation_thread)
        self.connect_threads_slots()
        self.calculation_thread.start()

        self.save_dataset_results_dir = DEFAULT_DATASET_RESULTS_FOLDER_NAME

        # Load the data
        self.raw_img_3d = nib.load('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz').get_data()

        self.current_slice = self.raw_img_3d[self.raw_img_3d.shape[0]//2 , : , :]
        self.current_slice_texture_results : np.ndarray
        self.current_slice_patchs_texture_results_arr = []

        self.dataset_path = 'E:\Projects\AmericanHeartTechnologies\AHT_TextureAnalysis\OneDrive_1_9-25-2024'
        self.dataset_analysis_settings_dict = {'num_rows': 3, 'num_cols':3,
                                                'available_texture_analysis_methods': DEFAULT_AVAILABLE_TEXTURE_ANALYSIS_METHODS.copy(),
                                                'save_path':self.save_dataset_results_dir,
                                                'radiomics_algs_settings':{'binWidth': 25, 'interpolator': sitk.sitkBSpline,'resampledPixelSpacing': None},
                                                '':''}

        self.patch_analysis_settings_dict = {'available_texture_analysis_methods': DEFAULT_AVAILABLE_TEXTURE_ANALYSIS_METHODS.copy(), 
                                                'radiomics_algs_settings':{'binWidth': 25, 'interpolator': sitk.sitkBSpline,'resampledPixelSpacing': None},
                                                '':''}
        
        # start the thread
        # self.texture_analyser.start()

    def connect_threads_slots(self):
        self.texture_analyser.analysis_of_wholeScan_started_sig.connect(self.new_analysis_in_dataset_started)
        self.texture_analyser.analysis_of_patch_finished_sig.connect(self.analysis_of_patch_finished_slot)
        self.texture_analyser.analysis_of_wholeScan_finished_sig.connect(self.analysis_of_one_scan_in_dataset_finished_slot)
        # self.texture_analyser.results_calculated_sig.connect(self.show_texture_analysis_results)
        self.start_singleImage_texture_analysis_sig.connect(self.texture_analyser.start_single_image_analysis_slot)
        self.start_dataset_texture_analysis_sig.connect(self.texture_analyser.start_datset_image_analysis_slot)
        
    def uiInitialize(self):
        self.setup_ui_connection()
        #self.ui.groupBox_3.setEnabled(False)
        #self.ui.groupBox_2.setEnabled(False)

    def setup_ui_connection(self):
        self.set_radio_btns_connection()
        self.set_lineEdits_connection()
        self.set_btns_connection()
        self.set_comboBoxs_connection()
        self.set_sliders_connection()
        self.set_checkBoxs_connection()
    
    def set_lineEdits_connection(self):
        self.ui.numRowsSpinBox.valueChanged.connect(self.on_numRowsChanged_changed)
        self.ui.numColumnsSpinBox.valueChanged.connect(self.on_numColsChanged_changed)
        pass #self.ui.velocityInput.textChanged.connect(lambda text: self.on_velocity_input_change(text))
        
    def set_btns_connection(self):
        self.ui.runAnalyzeWholeSliceBtn.clicked.connect(self.on_runAnalyzeWholeSliceBtn_clicked)
        self.ui.runAnalyzeSelectedPatchBtn.clicked.connect(self.on_runAnalyzeSelectedPatchBtn_clicked)
        self.ui.startDatasetAnalysisBtn.clicked.connect(self.on_start_dataset_analysis_btn_clicked)
        self.ui.exportSingleImgResultsToExcelBtn.clicked.connect(self.on_export_single_img_results_btn_clicked)
        self.ui.importNIFTIBtn.clicked.connect(self.on_import_single_NIFTI_btn_clicked)
        self.ui.selectDataDirBtn.clicked.connect(self.on_browse_dataset_path_btn_clicked)
        self.ui.browseOutputDirBtn.clicked.connect(self.on_browse_dataset_results_path_btn_clicked)

    def set_radio_btns_connection(self):
        pass #self.ui.MoveForDuration.clicked.connect(lambda: self.on_radio_btn_clicked(Mode.MOVE_IN_DURATION))
    
    def set_sliders_connection(self):
        self.ui.slicesSlider.valueChanged.connect(self.on_slices_slider_changed)
        

    def set_comboBoxs_connection(self):
        pass
    
    def set_checkBoxs_connection(self):
        #self.ui.medianFiltCheckBox.stateChanged.connect(self.on_medianCheckBox_toggled)
        pass

    def closeEvent(self, event):
        print ("close app")
        event.accept()
    
    def on_slices_slider_changed(self):
        try:
            self.ui.slicesSlider.setMaximum (self.raw_img_3d.shape[2]-1)
            self.current_slice = self.raw_img_3d[ : , : , self.ui.slicesSlider.value()]
            self.img_graphicsview.set_image_np(self.current_slice.copy() )
        except Exception as e:
            print(e)

    def multiSignalEmitsOnBtnClickedHandler(self):
        self.multiBtnSignalAvoider += 1
        if (self.multiBtnSignalAvoider < 3):
            return 0
        self.multiBtnSignalAvoider = 0
        return 1
    
    ##
    def convert_ndarray2qpixmap(self, ndarray_img):
        # Normalize the image to the range [0, 255]
        ndarray_img = ((ndarray_img - ndarray_img.min()) * (255 / (ndarray_img.max() - ndarray_img.min()))).astype('uint8')
        # Ensure the array is 2D for grayscale images
        if ndarray_img.ndim == 3:
            ndarray_img = ndarray_img[:, :, 0]

        height, width = ndarray_img.shape
        bytes_per_line = width * ndarray_img.itemsize
        qimage = QImage(ndarray_img.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)
    
    @pyqtSlot(dict)
    def show_texture_analysis_results(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.currentFrame = self.convert_cv_qt(cv_img)
        self.ui.liveImageLabel.setPixmap(self.currentFrame)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p) #(p)
    
    @pyqtSlot(dict)
    def new_analysis_in_dataset_started(self, dict):
        self.ui.numberOfProcessedFilesLabel.setText(f"Processing scan {dict['scan_number_in_dataset']+1} of {dict['total_scans']}" )
    
    def analysis_of_one_scan_in_dataset_finished_slot(self, dict): # receives a dict that includes: scan file name, scan number in dataset
        self.ui.datasetAnalysisProgressBar.setValue(int( 100 * (dict['scan_number_in_dataset']+1.0) / dict['total_scans']) )
    ##################  UI Components Slots  #################

    #ComboBoxs:
    def on_autoGainCombo_changed(self):
        pass
    
    #LineEdits:
    def on_camGainLineEdit_textChanged(self):
        pass

    def on_numRowsChanged_changed(self):
        self.img_graphicsview.set_num_rows(self.ui.numRowsSpinBox.value())
    
    def on_numColsChanged_changed(self):
        self.img_graphicsview.set_num_cols(self.ui.numColumnsSpinBox.value())

    #Buttons:
    def on_saveAsBtn_clicked(self):
        pass

    def on_runAnalyzeWholeSliceBtn_clicked(self):
        self.start_texture_analysis_sig.emit(self.current_slice, self.current_slice) # image, mask
        
        # self.current_slice_patchs_texture_results_arr.append(self.texture_analyser.runWholeAnalysis(img_2d=self.current_slice, mask_2d=self.current_slice))
        # for (key, val) in six.iteritems(self.current_slice_patchs_texture_results_arr[0]):
        #     print('  ', key, ':', val)
        #     self.ui.analyzeResultsListWidget.addItem(QListWidgetItem(str(key)+" : "+str(val)))
    
    def on_runAnalyzeSelectedPatchBtn_clicked(self):
        mask = np.zeros_like(self.img_graphicsview.numpy_image)
        mask[self.img_graphicsview.start_point_y : self.img_graphicsview.end_point_y, self.img_graphicsview.start_point_x : self.img_graphicsview.end_point_x] = 1
        # self.ui.label.setPixmap(self.convert_ndarray2qpixmap(self.img_graphicsview.numpy_image * mask))
        self.start_singleImage_texture_analysis_sig.emit(self.img_graphicsview.numpy_image, mask, self.patch_analysis_settings_dict)
    
    @pyqtSlot(dict)
    def analysis_of_patch_finished_slot(self, results_dict):
        self.ui.analyzeResultsListWidget.clear()
        for (key, val) in six.iteritems(results_dict):
            # print('  ', key, ':', val)
            self.ui.analyzeResultsListWidget.addItem(QListWidgetItem(str(key)+" : "+str(val)))

    def on_start_dataset_analysis_btn_clicked(self):
        self.start_dataset_texture_analysis_sig.emit(self.dataset_path, self.dataset_analysis_settings_dict)
        
    
    def on_export_single_img_results_btn_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, caption="Save File")
        self.texture_analyser.save_single_img_results(file_path)
            
    def on_import_single_NIFTI_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            try:
                self.raw_img_3d = nib.load(file_path).get_data()
                self.current_slice = self.raw_img_3d[:, :, self.raw_img_3d.shape[2]//2]
                self.ui.slicesSlider.setMaximum (self.raw_img_3d.shape[2]-1)
                self.ui.slicesSlider.setValue(self.raw_img_3d.shape[2]//2)
                # self.on_slices_slider_changed()

            except Exception as e:
                print (e)
            
    def on_browse_dataset_path_btn_clicked(self):
        load_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Load")
        if load_dir:
            # num_files = 0
            # for file_name in os.listdir(load_dir):
            #     file_path = os.path.join(load_dir, file_name)
            #     if os.path.isfile(file_path):
            #        num_files += 1
            # self.ui.inputDatasetDirLabel.setText(f"{len(os.listdir(load_dir))} scans loaded.\nDir: {load_dir}") #each scan files is in its own folder
            self.dataset_path = load_dir
            self.ui.inputDatasetDirLabel.setText(f"{len(os.listdir(load_dir))} files in directory.") #each scan files is in its own folder

    def on_browse_dataset_results_path_btn_clicked(self):
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if save_dir:
            self.ui.saveDatasetAnalysisResultsDir.setText(f"{save_dir}")
            self.save_dataset_results_dir = save_dir
            self.dataset_analysis_settings_dict['save_path'] = self.save_dataset_results_dir
    
    #checkbox:  
    def on_grayscaleCheckBox_toggled(self):
        pass

class GridGraphicsView(QGraphicsView):
    def __init__(self, rows, cols, parent=None):
        super(GridGraphicsView, self).__init__(parent)
        self.numpy_image : np.ndarray
        self.scene_qimage =  self.convert_ndarray2qImage(np.zeros([10, 10])) #QImage('t1_image.png')
        self.rows = 1
        self.cols = 1
        self.create_scene(self.scene_qimage, 2, 2)

    def set_image_np (self, np_img):
        self.numpy_image = np_img.copy() 
        self.scene_qimage = self.convert_ndarray2qImage(self.numpy_image)
        self.draw_image(self.scene_qimage)
        self.selected_patch = self.numpy_image.copy()

    def draw_image(self, img):
        try:
            self.scene_qimage = img
            pixmap = QPixmap.fromImage(self.scene_qimage.scaled(self.viewport().width(), self.viewport().height(), Qt.KeepAspectRatio))
            self.scene_img_qpixmapItem = self.scene.addPixmap(pixmap)
            # a : QGraphicsPixmapItem
            self.box_width = int (self.scene_img_qpixmapItem.boundingRect().width() // self.cols)
            self.box_height = int(self.scene_img_qpixmapItem.boundingRect().height() // self.rows)
            
        except Exception as e:
            print(e)

    def draw_grid(self, rows, cols):
        pen = QPen(QColor(Qt.black))
        self.rows = rows
        self.cols = cols
        for i in range(self.cols):
            for j in range(self.rows):
                rect = QGraphicsRectItem(QRectF(i * self.box_width, j * self.box_height, self.box_width, self.box_height))
                rect.setPen(pen)
                self.scene.addItem(rect)

    def mousePressEvent(self, event):
        point = self.mapToScene(event.pos())
        col = int(point.x()) // self.box_width
        row = int(point.y()) // self.box_height
        print (self.box_height)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            print(f'Clicked on row {row}, column {col}')
            if self.highlighted:
                self.scene.removeItem(self.highlighted)
            rect = QGraphicsRectItem(QRectF(col * self.box_width, row * self.box_height, self.box_width, self.box_height))
            print (row * self.box_height,row * self.box_height + self.box_height,col * self.box_width,col * self.box_width + self.box_width)

            self.start_point_x = int ( float(col * self.box_width) / self.scene_img_qpixmapItem.boundingRect().width() * self.numpy_image.shape[1])
            self.start_point_y = int ( float(row * self.box_height) / self.scene_img_qpixmapItem.boundingRect().height() * self.numpy_image.shape[0])
            self.end_point_x = int ( float( (col+1) * self.box_width) / self.scene_img_qpixmapItem.boundingRect().width() * self.numpy_image.shape[1])
            self.end_point_y = int ( float( (row+1) * self.box_height) / self.scene_img_qpixmapItem.boundingRect().height() * self.numpy_image.shape[0])
            self.selected_patch = self.numpy_image[self.start_point_y : self.end_point_y, self.start_point_x : self.end_point_x]
            rect.setPen(QPen(QColor(Qt.red)))
            self.scene.addItem(rect)
            self.highlighted = rect
    
    def set_num_rows(self, val):
        self.create_scene(self.scene_qimage, val, self.cols)

    def set_num_cols(self, val):
        self.create_scene(self.scene_qimage, self.rows, val)

    def create_scene(self, img, rows, cols):
        self.rows = rows
        self.cols = cols
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene_qimage = img
        self.box_width = int (self.scene_qimage.width() // self.cols)
        self.box_height = int (self.scene_qimage.height() // self.rows)
        self.highlighted = None
        self.draw_image(img)
        self.draw_grid(rows, cols)

    def convert_ndarray2qImage(self, ndarray_img):
        print(type(ndarray_img))
        ndarray_img = ((ndarray_img - ndarray_img.min()) * (1/(ndarray_img.max() - ndarray_img.min()) * 255)).astype('uint8')
        # Create QImage from the numpy array
        qimage = QImage(ndarray_img, ndarray_img.shape[1], ndarray_img.shape[0], QImage.Format_Grayscale8)
        # Convert QImage to QPixmap
        return qimage
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = mainwindow()
    a.show()
    sys.exit(app.exec_())
