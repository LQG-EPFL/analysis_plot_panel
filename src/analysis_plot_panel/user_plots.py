# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:57:44 2021

@author: Nick Sauerwein
"""

import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui

from __init__ import AnalysisPlot, color_palette



class ImagingPlot(AnalysisPlot):
    
    def __init__(self, title, **kwargs):
        
        super().__init__(title, **kwargs)
        
        self.setMinimumHeight(550)
        self.setMinimumWidth(550)
        
        
        self.axsumy = self.plots.addPlot(title="")
        self.axsumy.setFixedWidth(100)
        
        
        
        self.sumy = self.axsumy.plot()
        self.sumy_fit = self.axsumy.plot(pen=pg.mkPen(style=QtCore.Qt.DashLine, color = color_palette[1]))
        
        self.img = pg.ImageItem()
        self.aximg = self.plots.addPlot(title="")
        
        self.aximg.addItem(self.img)
        
        self.axsumy.setYLink(self.aximg)
        
        
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
        
        self.plots.nextRow()
        self.plots.nextColumn()
        
        self.axsumx = self.plots.addPlot()
        self.axsumx.setFixedHeight(100)
        self.axsumx.setXLink(self.aximg)
        
        
        
        self.sumx = self.axsumx.plot()
        self.sumx_fit = self.axsumx.plot(pen=pg.mkPen(style=QtCore.Qt.DashLine, color = color_palette[1]))
        
        
        self.table.setMinimumHeight(85)

        
        self.img.translate(-0.5, -0.5)
        
        self.scalex = 1
        self.scaley = 1
        
        self.cx = 0
        self.cy = 0
    
     
    def update(self, data_img, datax, datay, datax_fit, datay_fit, xgrid, ygrid, tabledata, warning):
        
        #update plots
        self.img.setImage(data_img.T)
        self.iso.setData(data_img.T)        
        
        self.data_img = data_img
        
        self.sumy.setData(datay, ygrid)
        self.sumy_fit.setData(datay_fit, ygrid)
        self.sumx.setData(xgrid, datax)
        self.sumx_fit.setData(xgrid, datax_fit)
        
        
        
        # set position and scale of image
        newscalex = xgrid[1] - xgrid[0]
        newscaley = ygrid[1] - ygrid[0]
        
        transx = (xgrid[0] - (self.cx - 0.5 * self.scalex)) / newscalex - 0.5
        transy = (ygrid[0] - (self.cy - 0.5 * self.scaley)) / newscaley - 0.5
        
        self.img.scale(newscalex/self.scalex, newscaley/self.scaley)
        self.img.translate(transx,transy)
        
        self.scalex = newscalex
        self.scaley = newscaley
        
        self.cx = xgrid[0]
        self.cy = ygrid[0]
        
        self.axsumx.setLabel('bottom', tabledata[0][0], units = 'mm')
        self.axsumy.setLabel('left', tabledata[1][0], units = 'mm')
        
        #update table and warning
        self.table.setData(tabledata)
        self.update_warning(warning)
         
    def updateIsocurve(self):
        self.iso.setLevel(self.isoLine.value())
        
    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.aximg.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data_img.shape[0] - 1))
        j = int(np.clip(j, 0, self.data_img.shape[1] - 1))
        val = self.data_img[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.aximg.setTitle("pos: (%0.1f, %0.1f) value: %g" % (x, y, val))


class MultiSpectrumPlot(AnalysisPlot):
    def __init__(self, title, labels, **kwargs):
        super().__init__(title, **kwargs)
        
        self.labels = labels
        
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        
        self.plot = self.plots.addPlot()
        
        self.curves_hist = {}
        self.curves_fit = {}
        for i, label in enumerate(labels):
           self.curves_hist[label]= self.plot.plot([0,1],[0], stepMode="center", fillLevel=0, fillOutline=True, brush=color_palette[i], name = label)
           self.curves_fit[label] = self.plot.plot(pen=pg.mkPen(style=QtCore.Qt.DashLine,width=0.5, color = (211,211,211), ))
        
        
        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot.graphicsItem())
        for i, label in enumerate(labels):
           self.legend.addItem(self.curves_hist[label], label)
           
        self.plot.setLabel('bottom', 'frequency', units = 'MHz')
        self.plot.setLabel('left', 'counts', units = '1')
        
        
    def update(self,data):
        for label in self.labels:
            freqs, counts, omega0, kappa, A, offset, f0, f1, duration,tabledata , warning = data[label]
            
            #update_plot
            self.freq2t = lambda freq: duration * (freq - f0)/(f1 - f0)
            self.cnt2rate = lambda c: c/(duration/(len(counts)))
            self.rate2cnt = lambda c: c*(duration/(len(counts)))
            
            deltafreqs = freqs[1] - freqs[0]
            freqs = np.append(freqs, freqs[-1] + deltafreqs)
            freqs -= deltafreqs / 2.
            
            self.curves_hist[label].setData(freqs, counts)
            
            
            lorenzian = lambda omega, omega0, kappa, A, offset: A * (kappa/2)**2 / ((omega - omega0)**2 + (kappa/2)**2) + offset
            freqsp = np.linspace(f0, f1, 400)
            
            self.curves_fit[label].setData(freqsp, lorenzian(2*np.pi*freqsp, omega0, kappa, A, offset))

class SpectrumPlot(AnalysisPlot):
    def __init__(self, title, maximal_count_rate = 10e6, **kwargs):
        super().__init__(title, **kwargs)
        
        
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        
        self.plot = self.plots.addPlot()
        
        self.curve_hist = self.plot.plot([0,1],[0], stepMode="center", fillLevel=0, fillOutline=True, brush=color_palette[0])
        self.curve_fit = self.plot.plot(pen=pg.mkPen(style=QtCore.Qt.DashLine,width=2, color = color_palette[1], ))
        
        self.plot.setLabel('bottom', 'frequency', units = 'MHz')
        self.plot.setLabel('left', 'counts', units = '1')
        
        self.maxrate = pg.InfiniteLine(angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        self.maximal_count_rate = maximal_count_rate
        
        self.plot.addItem(self.maxrate)
        
        f = lambda x: x
        
        self.secondary_xaxis('time', f, units = 's')
        self.secondary_yaxis('counte rate', f, units = 'Hz')
        
        
    def update(self,freqs, counts, omega0, kappa, A, offset, f0, f1, duration,tabledata , warning): 
        
        #update_plot
        self.freq2t = lambda freq: duration * (freq - f0)/(f1 - f0)
        self.cnt2rate = lambda c: c/(duration/(len(counts)))
        self.rate2cnt = lambda c: c*(duration/(len(counts)))
        
        self.axx2f = self.freq2t
        self.axy2f = self.cnt2rate
        
        self.maxrate.setValue(self.rate2cnt(self.maximal_count_rate))
        
        deltafreqs = freqs[1] - freqs[0]
        freqs = np.append(freqs, freqs[-1] + deltafreqs)
        freqs -= deltafreqs / 2.
        
        
        lorenzian = lambda omega, omega0, kappa, A, offset: A * (kappa/2)**2 / ((omega - omega0)**2 + (kappa/2)**2) + offset

        freqsp = np.linspace(f0, f1, 400)
        
        self.curve_hist.setData(freqs, counts)
        
        self.curve_fit.setData(freqsp, lorenzian(2*np.pi*freqsp, omega0, kappa, A, offset))
        
        
        #update table and warning
        self.table.setData(tabledata)
        self.update_warning(warning)
        
    def secondary_xaxis(self,label, f, **kwargs):
        
        self.axx2 = pg.AxisItem('top')
        self.axx2f = f
        
        self.plot.layout.addItem(self.axx2, 0 ,1)
        
        self.axx2.setLabel(label,**kwargs)
        
        def update_secondary_xaxis():
            view = np.array(self.plot.vb.viewRange()[0])
            self.axx2.setRange(*self.axx2f(view))
        
        self.plot.vb.sigXRangeChanged.connect(update_secondary_xaxis)
        self.plot.vb.sigResized.connect(update_secondary_xaxis)
        
    def secondary_yaxis(self,label, f,**kwargs):
        
        self.axy2 = pg.AxisItem('right')
        self.axy2f = f
        self.plot.layout.addItem(self.axy2, 2 ,2)
        
        self.axy2.setLabel(label,**kwargs)
        
        def update_secondary_yaxis():
            view = np.array(self.plot.vb.viewRange()[1])
            self.axy2.setRange(*self.axy2f(view))
        
        self.plot.vb.sigYRangeChanged.connect(update_secondary_yaxis)
        self.plot.vb.sigResized.connect(update_secondary_yaxis)