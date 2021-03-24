# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:07:54 2021

@author: nicks
"""

import lyse
import numpy as np

from uncertainties import ufloat

from data_extractors import DataExtractor

class FluoDataExtractor(DataExtractor):
    def __init__(self, imaging, cam, **kwargs):
        
        super().__init__(**kwargs)
        
        self.imaging = imaging
        self.cam = cam
                    
    def extract_data(self, h5_path , h5_file = None):
        
        run = lyse.Run(h5_path, no_write=True)
        
        imaging = self.imaging
        cam = self.cam
        
        try:
            from uncertainties import ufloat
            
            N = ufloat(*run.get_results(imaging, cam+'_Natoms', cam+'_Natoms_err', h5_file = h5_file))
            Nx = ufloat(*run.get_results(imaging, cam+'_Nx', cam+'_Nx_err', h5_file = h5_file))
            Ny = ufloat(*run.get_results(imaging, cam+'_Ny', cam+'_Ny_err', h5_file = h5_file))
            
            sx = ufloat(*run.get_results(imaging, cam+'_sx', cam+'_sx_err', h5_file = h5_file))
            sy = ufloat(*run.get_results(imaging, cam+'_sy', cam+'_sy_err', h5_file = h5_file))
            cx = ufloat(*run.get_results(imaging, cam+'_cx', cam+'_cx_err', h5_file = h5_file))
            cy = ufloat(*run.get_results(imaging, cam+'_cy', cam+'_cy_err', h5_file = h5_file))
            
            axesx, axesy = run.get_results(imaging, cam+'_axesx', cam+'_axesy')
            
            tabledata = np.array([
                (axesx,'{:.1ueS}'.format(Nx),'{:.1uS}'.format(sx),'{:.1uS}'.format(cx)), 
                (axesy, '{:.1ueS}'.format(Ny),'{:.1uS}'.format(sy),'{:.1uS}'.format(cy))
                ], dtype=[('axis',object),('N ('+'{:.1ueS}'.format(N)+')', object), ('c (mm)', object), ('s (mm)', object)])
    
            data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid = run.get_result_arrays(imaging, cam+'_diff_image',
                                                 cam+'_xsum',
                                                 cam+'_ysum',
                                                 cam+'_xfit', 
                                                 cam+'_yfit',
                                                 cam+'_xgrid',
                                                 cam+'_ygrid', h5_file = h5_file)
            
            warning = run.get_result(imaging, cam+'_warning')
        except:
            data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid, tabledata, warning = np.zeros((2,2)), np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(2),np.array([['no_data_x'],['no_data_y']]), 'no data in shot'
            

        return data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid, tabledata, warning
    
class AbsorptionDataExtractor(DataExtractor):
    def __init__(self, imaging, cam, **kwargs):
        
        super().__init__(**kwargs)
        
        self.imaging = imaging
        self.cam = cam
        
        
    def extract_data(self, h5_path, h5_file = None):
        
        run = lyse.Run(h5_path, no_write=True)
        
        imaging = self.imaging
        cam = self.cam
        try:
            N = run.get_result(imaging, cam+'_Natoms')
            Nx = ufloat(*run.get_results(imaging, cam+'_Nx', cam+'_Nx_err', h5_file = h5_file))
            Ny = ufloat(*run.get_results(imaging, cam+'_Ny', cam+'_Ny_err', h5_file = h5_file))
            
            sx = ufloat(*run.get_results(imaging, cam+'_sx', cam+'_sx_err', h5_file = h5_file))
            sy = ufloat(*run.get_results(imaging, cam+'_sy', cam+'_sy_err', h5_file = h5_file))
            cx = ufloat(*run.get_results(imaging, cam+'_cx', cam+'_cx_err', h5_file = h5_file))
            cy = ufloat(*run.get_results(imaging, cam+'_cy', cam+'_cy_err', h5_file = h5_file))
            
            axesx, axesy = run.get_results(imaging, cam+'_axesx', cam+'_axesy')
            
            tabledata = np.array([
                (axesx, '{:.1ueS}'.format(Nx),'{:.1uS}'.format(sx),'{:.1uS}'.format(cx)), 
                (axesy, '{:.1ueS}'.format(Ny),'{:.1uS}'.format(sy),'{:.1uS}'.format(cy))
                ], dtype=[('axis',object),('N ('+'{:.0f}'.format(N)+')', object), ('c (mm)', object), ('s (mm)', object)])
    
            data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid = run.get_result_arrays(imaging, cam+'_OD_image',
                                                 cam+'_xsum',
                                                 cam+'_ysum',
                                                 cam+'_xfit', 
                                                 cam+'_yfit',
                                                 cam+'_xgrid',
                                                 cam+'_ygrid', h5_file = h5_file)
            
            warning = run.get_result(imaging, cam+'_warning')
        except:
            data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid, tabledata, warning = np.zeros((2,2)), np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(2), np.arange(2),np.array([['no_data_x'],['no_data_y']]), 'no data in shot'
            
        return data_img, xsum, ysum, datax_fit, datay_fit, xgrid, ygrid, tabledata, warning

class SpectrumDataExtractor(DataExtractor):
    def __init__(self, name, frametype, **kwargs):
        
        super().__init__(**kwargs)
        
        self.name = name
        self.frametype = frametype
        
        
    def extract_data(self, h5_path, h5_file = None):
        
        run = lyse.Run(h5_path, no_write=True)
        
        imaging = 'cavity_spectrum'
        
        name = self.name
        frametype = self.frametype
        
        spectrum_name = name+' '+frametype
        
        try:
            
            f0 = run.get_result(imaging, spectrum_name+' f0', h5_file = h5_file)
            f1 = run.get_result(imaging, spectrum_name+' f1', h5_file = h5_file)
            duration = run.get_result(imaging, spectrum_name+' duration', h5_file = h5_file)
            
            kappa = run.get_result(imaging, spectrum_name+' kappa', h5_file = h5_file)
            omega0 = run.get_result(imaging, spectrum_name+' omega0', h5_file = h5_file)
            A = run.get_result(imaging, spectrum_name+' A', h5_file = h5_file)
            offset = run.get_result(imaging, spectrum_name+' offset', h5_file = h5_file)
            n_photons_total = run.get_result(imaging, spectrum_name+' n_photons_total', h5_file = h5_file)
            
            
            n_photons = run.get_result_array(imaging, spectrum_name+' n_photons', h5_file = h5_file)
            freqs = run.get_result_array(imaging, spectrum_name+' freqs', h5_file = h5_file)
            
            
            tabledata = np.array([
                ('{:.0f}'.format(n_photons_total),f'{omega0/2/np.pi:.3f}',f'{(kappa/2/np.pi):.3f}')
                ], dtype = [('N photons', object), ('omega0/2/pi (MHz)', object), ('kappa/2/pi (MHz)', object)])
            
            warning = run.get_result(imaging, spectrum_name+' warning')
        except:
            import traceback
            traceback.print_exc()
            freqs, n_photons, omega0, kappa, A, offset, f0, f1, duration,tabledata, warning = np.array([0.,1.]), np.array([0.,1.]), 0., 0., 0., 0.,0., 1, 1,np.arange(2,dtype = 'float64'), 'no data in shot'
            
        return freqs, n_photons, omega0, kappa, A, offset, f0, f1, duration, tabledata , warning