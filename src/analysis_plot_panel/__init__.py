# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:35:59 2021

@author: nicks
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from pyqtgraph.dockarea import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PIL import ImageColor

import lyse

import collections

from data_extractors import MultiDataExtractor, SingleDataExtractor, ArrayDataExtractor, EmptyDataExtractor, DataExtractorManager

color_palette_html = ['#1f77b4', 
                      '#ff7f0e', 
                      '#2ca02c', 
                      '#d62728', 
                      '#9467bd', 
                      '#8c564b',
                      '#e377c2', 
                      '#7f7f7f',
                      '#bcbd22',
                      '#17becf']

color_palette = [ImageColor.getcolor(color, "RGB") for color in color_palette_html]





from sortedcontainers import SortedSet

class ShotSelector(pg.LayoutWidget):
    
    valueChanged = pyqtSignal()
    selectionChanged = pyqtSignal()
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.nshots = 1
        
        self.setFixedHeight(100)
        
        self.current_idx_le= QLineEdit(self)
        
        self.current_idx_le.setMaximumWidth(30)
        self.current_idx_le.setValidator(QtGui.QIntValidator()) 
        
        self.current_idx_le.setText(str(-1))
        
        self.addWidget(self.current_idx_le)
        
        self.slider = QSlider(self)
        
        self.slider.setPageStep(1)
        self.slider.setOrientation(Qt.Horizontal)
        
        self.addWidget(self.slider, colspan = 2)
        
        self.nextRow()
        
        self.addWidget(QLabel('index selector'))
        
        self.idx_select_le = QLineEdit(self)
        self.idx_select_le.setText(':')
        self.addWidget(self.idx_select_le)
        
        self.warning = QLabel()
        self.warning.setMargin(5)
        self.update_warning()
        self.addWidget(self.warning)
        
        self.update_nshots(self.nshots)
        
        self.idx_select_le.editingFinished.connect(self.update_selection)
        self.current_idx_le.editingFinished.connect(self.setSliderValue)
        self.slider.valueChanged.connect(self.setLabelValue)
    
    def update_nshots(self, nshots):
        self.nshots = nshots
        self.idx = np.arange(self.nshots)
        self.slider.setRange(0, self.nshots - 1)
        
        self.update_selection()
        self.setSliderValue()

    
    def update_warning(self, warning = ''):
        if warning == '':
            self.warning.setStyleSheet("background-color: lightgreen")
            warning = 'all good'
        else:
            self.warning.setStyleSheet("background-color: red")
        
        self.warning.setText(warning)
        
    def update_selection(self):
        self.update_warning()
        
        slice_text = self.idx_select_le.text()
        slices = slice_text.split(',')
        
        self.idx_selected = SortedSet([])
        for s in slices:
            try:
                scope = locals()
                
                select = eval('self.idx['+s+']', scope)
                
                if isinstance(select, np.ndarray):
                    for idx in select:
                        self.idx_selected.add(idx)
                else:
                    self.idx_selected.add(select)
                    
            except:
                self.update_warning('problem in selected indeces')
                return 0
            
        self.slider.setRange(0, len(self.idx_selected) - 1)
        
        
        if int(self.current_idx_le.text())%self.nshots not in self.idx_selected:
            self.current_idx_le.setText(self.idx_selected[-1])
            
            self.update_warning('last index not in selection <br> -> setting last selected')
            
        self.selectionChanged.emit()
            
    def setLabelValue(self, value):
        newval = self.idx_selected[value]
        
        if newval != self.get_current_index():
            self.current_idx_le.setText(str(newval))
            
            self.valueChanged.emit()
        
    def setSliderValue(self):
        self.update_warning()
        
        value = int(self.current_idx_le.text())
        
        try:
            value_sl = self.idx_selected.index(value%len(self.idx))
            self.slider.setValue(value_sl)
        except ValueError:
            self.update_warning('set index not in selection <br> ignore')
    
    def get_current_index(self):
        return int(self.current_idx_le.text())%self.nshots
    
    def get_selected_indices(self):
        return (np.array(self.idx_selected),)
            
        
class AnalysisPlotPanel(QtGui.QMainWindow):
    
    def __init__(self, h5_paths,**kwargs):
        
        self.h5_paths = h5_paths
        
        super().__init__(**kwargs)
        
        pg.mkQApp()
        
        
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.area = DockArea()
        
        self.setCentralWidget(self.area)
        self.resize(1000,500)
        
        self.dshotselector = Dock("Shot selector")
        self.shotselector = ShotSelector()
        
        self.dshotselector.addWidget(self.shotselector)
        self.area.addDock(self.dshotselector, 'bottom')
        
        self.qpg_dock = Dock("Quick Plot Generator")
        self.qpg_dock.addWidget(QuickPlotGenerator(self))
        self.qpg_dock.setMinimumSize(self.qpg_dock.minimumSizeHint())
        self.area.addDock(self.qpg_dock)
        
        
        self.show()
        
        self.plots = {}
        self.data_extractor_manager = DataExtractorManager()
        
        self.shotselector.valueChanged.connect(self.refresh)
        self.shotselector.selectionChanged.connect(self.refresh)
        
        self.df = lyse.data()
        
    def add_plot_dock(self, plot_name,plot_widget, data_extractor, **kwargs):
        if plot_name not in self.plots:
            plot_widget.plot_name = plot_name
            plot_widget.data_extractor = data_extractor
            self.data_extractor_manager[plot_name] = data_extractor
            
            dock = Dock(plot_name,**kwargs)
            
            dock.sigClosed.connect(self.remove_plot)
            dock.addWidget(plot_widget)
            
            self.area.addDock(dock, 'right')
            
            self.plots[plot_name] = plot_widget
            
            
        else:
            print (f'Plot {plot_name} already exists. Please choose different name.')
    
    def remove_plot(self ,dock):
        del self.plots[dock.title()]
        
    def update_h5_paths(self, h5_paths):
        self.h5_paths = h5_paths
        
        self.shotselector.update_nshots(len(h5_paths))
        
        for plot_name, plot in self.plots.items():
            plot.data_extractor.clean_memory(h5_paths)
            
    def refresh(self, h5_path = None):
        if len(self.h5_paths):

            self.h5_paths_selected = self.h5_paths.iloc[self.shotselector.get_selected_indices()]
            
            
            if h5_path == None:
                i = self.shotselector.get_current_index()
                h5_path = self.h5_paths.iloc[i]
                
            
            self.data_extractor_manager.update_local_data(h5_path)
            
            for plot_name, plot in self.plots.items():
                plot.update_from_h5(h5_path)
        else:
            pass


class DataPlot(QSplitter):
    def __init__(self,  **kwargs):
        
        super().__init__(**kwargs)
        
        self.setOrientation(Qt.Vertical)
        
        self.plots = pg.GraphicsLayoutWidget()
        
        self.addWidget(self.plots)
        
        self.bottom = QSplitter()
        self.bottom.setOrientation(Qt.Horizontal)
        
        self.addWidget(self.bottom)
        
        self.h5_path_shown = None
        
    def update_from_h5(self, h5_path):
        if self.h5_path_shown != h5_path or self.data_extractor.local_data_changed:
            self.h5_path_shown = h5_path
            self.update(*self.data_extractor.extract_data(h5_path))
        
class QuickDataPlot(DataPlot):
    def __init__(self, ap, **kwargs):
        
        super().__init__(**kwargs)
        
        self.plot = self.plots.addPlot()
        
        for key in self.plot.axes:
            ax = self.plot.getAxis(key)
            # Fix Z value making the grid on top of the image
            ax.setZValue(1)
        self.ap = ap
        self.h5_paths_shown = []
        
        self.table = QTableWidget()
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        
        self.bottom.addWidget(self.table)
        
        self.plot_setting = PlotSettings(self.plot)
        
        self.bottom.addWidget(self.plot_setting)
        
    def update_from_h5(self, h5_path = None):
        self.update_data_extractor()
        if self.data_extractor.local_data_changed or collections.Counter(self.h5_paths_shown) != collections.Counter(self.ap.h5_paths_selected):
            self.h5_paths_shown = self.ap.h5_paths_selected
            self.update()


class AnalysisPlot(DataPlot):
    def __init__(self, title,  **kwargs):
        
        super().__init__(**kwargs)
        
        
        self.table = pg.TableWidget()
        
        self.bottom.addWidget(self.table)
        
        self.desciption = pg.LayoutWidget()
        self.title = QtGui.QLabel()
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setText('<h2>'+title+' <\h2>')
        self.warning = QtGui.QLabel()
        self.warning.setAlignment(QtCore.Qt.AlignCenter)
        
        self.desciption.addWidget(self.title)
        self.desciption.nextRow()
        self.desciption.addWidget(self.warning)
        
        self.bottom.addWidget(self.desciption)
        
    def update_warning(self, warning):
        if warning == '':
            self.warning.setStyleSheet("background-color: lightgreen")
            warning = 'all good'
        else:
            self.warning.setStyleSheet("background-color: red")
        
        self.warning.setText(warning)
        
    def update_from_h5(self, h5path):
        if hasattr(self, 'data_extractor'):
            self.update(*self.data_extractor.get_data(h5path))
        else:
            pass

class ExtendedCombo( QComboBox ):
    def __init__( self,  parent = None):
        super().__init__( parent )

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable( True )
        self.completer = QCompleter( self )

        # always show all completions
        self.completer.setCompletionMode( QCompleter.UnfilteredPopupCompletion )
        self.pFilterModel = QSortFilterProxyModel( self )
        self.pFilterModel.setFilterCaseSensitivity( Qt.CaseInsensitive )

        self.completer.setPopup( self.view() )

        self.setCompleter( self.completer )

        self.lineEdit().textEdited[str].connect( self.pFilterModel.setFilterFixedString )
        self.completer.activated.connect(self.setTextIfCompleterIsClicked)

    def setModel( self, model ):
        super().setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    def setModelColumn( self, column ):
        self.completer.setCompletionColumn( column )
        self.pFilterModel.setFilterKeyColumn( column )
        super().setModelColumn( column )


    def view( self ):
        return self.completer.popup()

    def index( self ):
        return self.currentIndex()

    def setTextIfCompleterIsClicked(self, text):
      if text:
        index = self.findText(text)
        self.setCurrentIndex(index)


class PlotSettings(QTableWidget):
    def __init__(self,plot, **kwargs):
        super().__init__(**kwargs)
        
        self.plot = plot
        self.setColumnCount(3)
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 100)
        
        self.setHorizontalHeaderLabels(['parameter', 'setting', 'units'])
        
        self.setRowCount(4)
        
        
        # title
        i = 0
        self.le_title = QLineEdit()
        
        self.setCellWidget(i, 0, QLabel('title'))
        self.setCellWidget(i, 1, self.le_title)
        
        self.le_title.textChanged[str].connect(self.set_title)
        
        
        # xlabel
        i +=1
        self.le_xlabel = QLineEdit()
        self.le_xlabel_unit = QLineEdit()
        
        self.setCellWidget(i, 0, QLabel('xlabel'))
        self.setCellWidget(i, 1, self.le_xlabel)
        self.setCellWidget(i, 2, self.le_xlabel_unit)
        
        self.le_xlabel.textChanged[str].connect(self.set_xlabel)
        self.le_xlabel_unit.textChanged[str].connect(self.set_xlabel)
        
        # ylabel
        i +=1
        self.le_ylabel = QLineEdit()
        self.le_ylabel_unit = QLineEdit()
        
        self.setCellWidget(i, 0, QLabel('ylabel'))
        self.setCellWidget(i, 1, self.le_ylabel)
        self.setCellWidget(i, 2, self.le_ylabel_unit)
        
        self.le_ylabel.textChanged[str].connect(self.set_ylabel)
        self.le_ylabel_unit.textChanged[str].connect(self.set_ylabel)
        
        # grid
        i +=1
        self.cb_grid = QCheckBox()
        self.setCellWidget(i, 0, QLabel('grid'))
        self.setCellWidget(i, 1, self.cb_grid)
        
        self.cb_grid.stateChanged.connect(self.set_grid)
        
    def set_title(self):
        self.plot.setTitle(self.le_title.text())
        
    def set_xlabel(self):
        self.plot.setLabel('bottom', self.le_xlabel.text(), units = self.le_xlabel_unit.text())
        
    def set_ylabel(self):
        self.plot.setLabel('left', self.le_ylabel.text(), units = self.le_ylabel_unit.text())
        
    def set_grid(self):
        self.plot.showGrid(x = self.cb_grid.isChecked(), y = self.cb_grid.isChecked(), alpha = 0.3)
  
from pandas.api.types import is_numeric_dtype
class NumericDataCombo(ExtendedCombo):
    
    def __init__(self, df, **kwargs):
        
        super().__init__(**kwargs)
        self.update_model(df)
                
    def update_model(self, df):
        model = QtGui.QStandardItemModel()
        
        
        
        item = QtGui.QStandardItem('shot number')
        model.setItem(0, 0, item)
        
        i = 1
        for midx in df.columns:
            if is_numeric_dtype(df.dtypes[midx]):
                column_name = ','.join([x for x in midx if x])
                item = QtGui.QStandardItem(column_name)
                model.setItem(i, 0, item)
                i+=1
                
        self.setModel(model)
        
    def get_idx(self):
        return tuple(str(self.currentText()).split(','))
        
class Quick1DPlot(QuickDataPlot):
    
    def __init__(self,ap,**kwargs):
        
        super().__init__(ap,**kwargs)        
        self.nplots = 0
        
        self.table.setColumnCount(5)
        self.table.setRowCount(1)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 30)
        self.table.setColumnWidth(3, 30)
        self.table.setColumnWidth(4, 40)
        self.table.setHorizontalHeaderLabels(['xvalue', 'yvalue', 'color', 'show', 'scatter'])
         
        self.combos = []
        self.curves = []
        self.show_cbs = []
        self.scatter_cbs = []
        
        self.mk_buttons()
        
            
    def mk_buttons(self):
        self.bt_add_plot = QPushButton('Add plot', self)
        self.bt_add_plot.clicked.connect(self.add_plot)
        self.table.setCellWidget(self.nplots, 0, self.bt_add_plot)
        
        
        self.bt_update = QPushButton('Update', self)
        self.bt_update.clicked.connect(self.update_from_h5)
        self.table.setCellWidget(self.nplots, 1, self.bt_update)
    
    def add_plot(self):
        self.nplots += 1
        self.table.setRowCount(self.nplots+1)
        combox = NumericDataCombo(self.ap.df)
        comboy = NumericDataCombo(self.ap.df)
        
        self.combos += [[combox,comboy]]
        self.table.setCellWidget(self.nplots - 1, 0, combox)
        self.table.setCellWidget(self.nplots - 1, 1, comboy)
        
        self.table.setItem(self.nplots - 1, 2, QtGui.QTableWidgetItem())
        self.table.item(self.nplots - 1, 2).setBackground(QtGui.QColor(*color_palette[self.nplots - 1]))
        
        #self.table.setFixedSize(self.table.sizeHint())
        
        self.curves += [self.plot.plot(pen=pg.mkPen(color = color_palette[self.nplots - 1], width = 1.5), symbol ='x', symbolPen = None, symbolBrush = None)]
        
        
        self.show_cbs += [QCheckBox()]
        self.show_cbs[self.nplots - 1].setChecked(True)
        self.table.setCellWidget(self.nplots - 1, 3, self.show_cbs[self.nplots - 1])
        self.show_cbs[self.nplots - 1].stateChanged.connect(self.update_shows)
        
        self.scatter_cbs += [QCheckBox()]
        self.scatter_cbs[self.nplots - 1].setChecked(False)
        self.table.setCellWidget(self.nplots - 1, 4, self.scatter_cbs[self.nplots - 1])
        self.scatter_cbs[self.nplots - 1].stateChanged.connect(self.update_scatters)
        
        self.mk_buttons()
    
        
    def update_shows(self):
        for k, cb in enumerate(self.show_cbs):
            if cb.isChecked():
                self.curves[k].show()
            else:
                self.curves[k].hide()
                
    def update_scatters(self):
        for k, cb in enumerate(self.scatter_cbs):
            pen=pg.mkPen(color = color_palette[k], width = 1.5)
            brush=pg.mkBrush(color = color_palette[k])
            if cb.isChecked():
                self.curves[k].setSymbolBrush(brush)
                self.curves[k].setPen(None)
            else:
                self.curves[k].setSymbolBrush(None)
                self.curves[k].setPen(pen)

        
    def update_data_extractor(self):
        
        idxxs = [combo[0].get_idx() for combo in self.combos]
        idxys = [combo[1].get_idx() for combo in self.combos]
        
        for idxx in idxxs:
            if idxx not in self.data_extractor.data_extractors:
                if idxx[0] == 'shot number':
                    self.data_extractor[idxx] = EmptyDataExtractor()
                else:
                    self.data_extractor[idxx] = SingleDataExtractor(idxx)
                    
        for idxy in idxys:
            if idxy not in self.data_extractor.data_extractors:
                if idxy[0] == 'shot number':
                    self.data_extractor[idxy] = EmptyDataExtractor()
                else:
                    self.data_extractor[idxy] = SingleDataExtractor(idxy)
        
        self.data_extractor.clean_children(idxxs + idxys)
        self.data_extractor.clean_memory(self.ap.h5_paths)
        
    def update(self, data = None):
        
        Xs = np.zeros((self.nplots, len(self.ap.h5_paths_selected)))
        Ys = np.zeros((self.nplots, len(self.ap.h5_paths_selected)))
        
        

        for i, h5_path in enumerate(self.ap.h5_paths_selected):
            
            data = self.data_extractor.get_data(h5_path)[0]
            
            for k in range(self.nplots):
                idxx = self.combos[k][0].get_idx()
                idxy = self.combos[k][1].get_idx()
                
                if idxx[0] == 'shot number':
                    Xs[k, i] =  self.ap.shotselector.get_selected_indices()[0][i]
                else:
                    Xs[k, i] = data[idxx]
                    
                if idxy[0] == 'shot number':
                    Ys[k, i] =  self.ap.shotselector.get_selected_indices()[0][i]
                else:
                    Ys[k, i] = data[idxy]
                    
        for k in range(self.nplots):
            self.curves[k].setData(Xs[k], Ys[k])

import h5py  
from pandas.api.types import is_numeric_dtype
class ArrayDataCombo(ExtendedCombo):
    
    def __init__(self, h5_paths, **kwargs):
        
        super().__init__(**kwargs)
        self.update_model(h5_paths)
                
    def update_model(self, h5_paths):
        results_array_labels = set([])
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as h5_file:
                analysis_names = h5_file['results'].keys()
                
                for analysis_name in analysis_names:
                    for key in h5_file['results'][analysis_name].keys():
                        results_array_labels.add(analysis_name+','+key)
            
        
        model = QtGui.QStandardItemModel()

        item = QtGui.QStandardItem('shot number')
        model.setItem(0, 0, item)
        
        
        for i, idx in enumerate(results_array_labels):
            item = QtGui.QStandardItem(idx)
            model.setItem(i, 0, item)
                
        self.setModel(model)
        
    def get_idx(self):
        return tuple(str(self.currentText()).split(','))

class QuickWaterfallPlot(QuickDataPlot):
    
    def __init__(self,*args,**kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        
        # Isocurve draplotsg
        self.iso = pg.IsocurveItem(level=1000, pen=color_palette[2])
        self.iso.setParentItem(self.img)
        self.iso.setZValue(5)
        
        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.plots.addItem(self.hist)
        
        # Draggable line for setting isocurve level
        self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen=color_palette[2])
        self.hist.vb.addItem(self.isoLine)
        self.hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        self.isoLine.setValue(1000)
        self.isoLine.setZValue(1000) # bring iso line above contrast controls
        self.isoLine.sigDragged.connect(self.updateIsocurve)

    

        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = self.imageHoverEvent
        
        self.img.translate(-0.5, -0.5)
        
        self.scalex = 1
        self.scaley = 1
        
        self.cx = 0
        self.cy = 0
        
        self.nplots = 0

        self.table.setColumnCount(3)
        self.table.setRowCount(2)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 150)
        self.table.setHorizontalHeaderLabels(['xvalue', 'yarray', 'zarray'])
        
        self.combox = NumericDataCombo(self.ap.df)
        self.comboy = ArrayDataCombo(self.ap.h5_paths)
        self.comboz = ArrayDataCombo(self.ap.h5_paths)
        
        self.table.setCellWidget(0, 0, self.combox)
        self.table.setCellWidget(0, 1, self.comboy)
        self.table.setCellWidget(0, 2, self.comboz)
        
        self.mk_buttons()      
            
    def mk_buttons(self):
        self.bt_update = QPushButton('Update', self)
        self.bt_update.clicked.connect(self.update_from_h5)
        self.table.setCellWidget(1, 1, self.bt_update)
        
    def update_data_extractor(self):
        
        idxx = self.combox.get_idx()
        idxy = self.comboy.get_idx()
        idxz = self.comboz.get_idx()
        
        if idxx not in self.data_extractor.data_extractors:
            if idxx[0] == 'shot number':
                self.data_extractor[idxx] = EmptyDataExtractor()
            else:
                self.data_extractor[idxx] = SingleDataExtractor(idxx)
        
        if idxy not in self.data_extractor.data_extractors:
            self.data_extractor[idxy] = ArrayDataExtractor(idxy)
            
        if idxz not in self.data_extractor.data_extractors:
            self.data_extractor[idxz] = ArrayDataExtractor(idxz)
        
        self.data_extractor.clean_children([idxx, idxy, idxz])
        self.data_extractor.clean_memory(self.ap.h5_paths)
        
    def update(self, data = None):
        xs = np.array([])
        ys = np.array([])
        zs = np.array([])
        
        idxx = self.combox.get_idx()
        idxy = self.comboy.get_idx()
        idxz = self.comboz.get_idx()
        
        for i, h5_path in enumerate(self.ap.h5_paths_selected):
            
            data = self.data_extractor.get_data(h5_path)[0]
            
            
            if idxx[0] == 'shot number':
                x = self.ap.shotselector.get_selected_indices()[0][i]
            else:
                x = data[idxx]
            
            y = data[idxy]
            z = data[idxz]
            
            if y is not None and z is not None:
                xs = np.append(xs , x * np.ones_like(y))
                ys = np.append(ys , y)
                zs = np.append(zs , z)
            elif y is not None:
                xs = np.append(xs , x * np.ones_like(y))
                ys = np.append(ys , y)
                zs = np.append(zs , y*np.nan)
            elif z is not None:
                xs = np.append(xs , x * np.ones_like(z))
                ys = np.append(ys , z*np.nan)
                zs = np.append(zs , z)
           
        # here we don't want to assume that the data is on a grid
        # this can happen if the parameters where changed between two sweeps
        xi = np.linspace(xs.min(), xs.max(), 200)
        yi = np.linspace(ys.min(), ys.max(), 200)
        
        Xi, Yi = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        
        Zi = griddata((xs,ys), zs, (Xi, Yi), 'nearest')
        
        
        
        
        
        self.img.setImage(Zi.T)
        self.iso.setData(Zi.T)  
        self.data_img = Zi
        
        # set position and scale of image
        newscalex = xi[1] - xi[0]
        newscaley = yi[1] - yi[0]
        
        transx = (xi[0] - (self.cx - 0.5 * self.scalex)) / newscalex - 0.5
        transy = (yi[0] - (self.cy - 0.5 * self.scaley)) / newscaley - 0.5
        
        self.img.scale(newscalex/self.scalex, newscaley/self.scaley)
        self.img.translate(transx,transy)
        
        self.scalex = newscalex
        self.scaley = newscaley
        
        self.cx = xi[0]
        self.cy = yi[0]
        
    def updateIsocurve(self):
        self.iso.setLevel(self.isoLine.value())
        
    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.plot.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data_img.shape[0] - 1))
        j = int(np.clip(j, 0, self.data_img.shape[1] - 1))
        val = self.data_img[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.plot.setTitle("pos: (%0.1f, %0.1f) value: %g" % (x, y, val))

class Quick2DPlot(QuickDataPlot):
    
    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        
        # Isocurve draplotsg
        self.iso = pg.IsocurveItem(level=1000, pen=color_palette[2])
        self.iso.setParentItem(self.img)
        self.iso.setZValue(5)
        
        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.plots.addItem(self.hist)
        
        # Draggable line for setting isocurve level
        self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen=color_palette[2])
        self.hist.vb.addItem(self.isoLine)
        self.hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        self.isoLine.setValue(1000)
        self.isoLine.setZValue(1000) # bring iso line above contrast controls
        self.isoLine.sigDragged.connect(self.updateIsocurve)

    

        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = self.imageHoverEvent
        
        self.img.translate(-0.5, -0.5)
        
        self.scalex = 1
        self.scaley = 1
        
        self.cx = 0
        self.cy = 0
        
        self.nplots = 0
        
        self.table.setColumnCount(3)
        self.table.setRowCount(2)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 150)
        self.table.setHorizontalHeaderLabels(['xvalue', 'yarray', 'zarray'])
        
        self.combox = NumericDataCombo(self.ap.df)
        self.comboy = NumericDataCombo(self.ap.df)
        self.comboz = NumericDataCombo(self.ap.df)
        
        self.table.setCellWidget(0, 0, self.combox)
        self.table.setCellWidget(0, 1, self.comboy)
        self.table.setCellWidget(0, 2, self.comboz)
        
        self.mk_buttons()
            
    def mk_buttons(self):
        self.bt_update = QPushButton('Update', self)
        self.bt_update.clicked.connect(self.update_from_h5)
        self.table.setCellWidget(1, 1, self.bt_update)
    def update_data_extractor(self):
        
        idxx = self.combox.get_idx()
        idxy = self.comboy.get_idx()
        idxz = self.comboz.get_idx()
        
        if idxx not in self.data_extractor.data_extractors:
            if idxx[0] == 'shot number':
                self.data_extractor[idxx] = EmptyDataExtractor()
            else:
                self.data_extractor[idxx] = SingleDataExtractor(idxx)   
                
        if idxy not in self.data_extractor.data_extractors:
            if idxy[0] == 'shot number':
                self.data_extractor[idxy] = EmptyDataExtractor()
            else:
                self.data_extractor[idxy] = SingleDataExtractor(idxy)  
        
        if idxz not in self.data_extractor.data_extractors:
            if idxz[0] == 'shot number':
                self.data_extractor[idxz] = EmptyDataExtractor()
            else:
                self.data_extractor[idxz] = SingleDataExtractor(idxz)
                
        self.data_extractor.clean_children([idxx, idxy, idxz])
        self.data_extractor.clean_memory(self.ap.h5_paths)
        
    def update(self, murks = None):
        
        xs = np.array([])
        ys = np.array([])
        zs = np.array([])
        
        idxx = self.combox.get_idx()
        idxy = self.comboy.get_idx()
        idxz = self.comboz.get_idx()
        
        for i, h5_path in enumerate(self.ap.h5_paths_selected):
            
            data = self.data_extractor.get_data(h5_path)[0]
            if idxx[0] == 'shot number':
                xs = np.append(xs, self.ap.shotselector.get_selected_indices()[0][i])
            else:
                xs = np.append(xs, data[idxx])
                
            if idxy[0] == 'shot number':
                ys = np.append(ys, self.ap.shotselector.get_selected_indices()[0][i])
            else:
                ys = np.append(ys, data[idxy])
                
            if idxz[0] == 'shot number':
                zs = np.append(zs, self.ap.shotselector.get_selected_indices()[0][i])
            else:
                zs = np.append(zs, data[idxz])
        
        
        # here we don't want to assume that the data is on a grid
        # this can happen if the parameters where changed between two sweeps
        
        xi = np.linspace(xs.min(), xs.max(), 200)
        yi = np.linspace(ys.min(), ys.max(), 200)
        
        Xi, Yi = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        
        Zi = griddata((xs,ys), zs, (Xi, Yi), 'nearest')
        
        self.img.setImage(Zi.T)
        self.iso.setData(Zi.T)  
        self.data_img = Zi
        
        # set position and scale of image
        newscalex = xi[1] - xi[0]
        newscaley = yi[1] - yi[0]
        
        transx = (xi[0] - (self.cx - 0.5 * self.scalex)) / newscalex - 0.5
        transy = (yi[0] - (self.cy - 0.5 * self.scaley)) / newscaley - 0.5
        
        self.img.scale(newscalex/self.scalex, newscaley/self.scaley)
        self.img.translate(transx,transy)
        
        self.scalex = newscalex
        self.scaley = newscaley
        
        self.cx = xi[0]
        self.cy = yi[0]
        
        
        
        
    def updateIsocurve(self):
        self.iso.setLevel(self.isoLine.value())
        
    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.plot.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data_img.shape[0] - 1))
        j = int(np.clip(j, 0, self.data_img.shape[1] - 1))
        val = self.data_img[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.plot.setTitle("pos: (%0.1f, %0.1f) value: %g" % (x, y, val))
  
    
class QuickPlotGenerator(pg.LayoutWidget):
    
    def __init__(self, ap, **kwargs):
        super().__init__(**kwargs)
        
        self.ap = ap
        
        self.title = QLabel('<h2> Quick Plot Generator <\h2>')
        
        self.addWidget(self.title,colspan = 2)
        
        self.nextRow()
        
        self.newlayout = pg.LayoutWidget()
        
        self.newlayout.addWidget(QLabel('Make new plot'))
        self.newlayout.nextRow()
        self.newlayout.addWidget(QLabel('Title: '))
        
        self.title_le = QLineEdit('murks')
        self.newlayout.addWidget(self.title_le)
        
        self.newlayout.nextRow()
        self.bt1d = QPushButton('make 1D plot', self)
        self.newlayout.addWidget(self.bt1d)
        
        self.bt1d.clicked.connect(self.mk1dplot)
        
        self.btwaterfall = QPushButton('make waterfall plot', self)
        self.newlayout.addWidget(self.btwaterfall)
        
        self.btwaterfall.clicked.connect(self.mkwaterfallplot)
        
        self.bt2d = QPushButton('make 2d plot', self)
        self.newlayout.addWidget(self.bt2d)
        
        self.bt2d.clicked.connect(self.mk2dplot)
        
        self.addWidget(self.newlayout)
        
    def mk1dplot (self):
        title = self.title_le.text()
        self.ap.add_plot_dock(title, Quick1DPlot(self.ap),  MultiDataExtractor(), closable = True)
        
    def mkwaterfallplot (self):
        title = self.title_le.text()
        self.ap.add_plot_dock(title, QuickWaterfallPlot(self.ap),  MultiDataExtractor(), closable = True)
        
    def mk2dplot (self):
        title = self.title_le.text()
        self.ap.add_plot_dock(title, Quick2DPlot(self.ap),  MultiDataExtractor(), closable = True)
        
        
        
        
        
        
        
        