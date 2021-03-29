
import lyse
import time
import h5py
import numpy as np
import matplotlib
# import lqg_utils as lqg
# matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib import gridspec
from pathlib import Path
from scipy.ndimage import gaussian_filter
from uncertainties import ufloat
from uncertainties.umath import *
import matplotlib.patches as patches  

from skimage.transform import rotate
#################################################################
# Load h5
#################################################################

# singleshot_data = data(path)
# Is this script being run from within an interactive lyse session?
if lyse.spinning_top:
    # If so, use the filepath of the current shot
    h5_path = lyse.path
else:
    # If not, get the filepath of the last shot of the lyse DataFrame
    df = lyse.data()
    h5_path = df.filepath.iloc[-1]

# Instantiate a lyse.Run object for this shot
run = lyse.Run(h5_path)


debug= False#run_globals['debug_abs_imaging']
use_matplotlib = False

#################################################################
# Physics constants
#################################################################
hbar = 1.0545718e-34 #J.s
c    = 299792458 #m.s-1
D2_frequency  = 446.799677e12 #Hz
D2_wavelength = 670.977338e-9 #m
D2_linewidth  = 36.898e6 #s-1
D2_Isat = 2.54e-3 #W.cm-2
m6Li    = 9.988341e-27 #kg
kB      = 1.38064852e-23 #m2.kg.s-1.K-1

OD_cap = 1
#################################################################
# Experimental parameters
#################################################################

D2_detuning = 0 #Hz
I_in = 0.035*D2_Isat #W.cm-2




# Get a dictionary of the global variables used in this shot
run_globals = run.get_globals()




cam_names = []
if run_globals['absorption_imaging']:
    cam_names +=['Cam_absorption']

run.save_result('cams', cam_names)

pixel_size = {
    'Cam_absorption' : 3.45e-6
}
magnification = {
    'Cam_absorption' : 4.
}

npx = {
    'Cam_absorption' : [1288, 964]
}

tof = {
    'Cam_absorption' : run_globals['abs_tof']
}

cam_axes = {
    'Cam_absorption' : {'x' : 'y' , 'y' : r'x'} ,
    }

angle = {
    'Cam_absorption' : 7.5
    }

effective_pixel_size = {
    name : pixel_size[name] * magnification[name]
    for name in cam_names
}

pixel_area = {
    name : (effective_pixel_size[name])**2
    for name in cam_names
}

px_offsets = {}

if run_globals['abs_crop_hardware']:
    px_offsets['Cam_absorption'] =  (run_globals['abs_hard_ROI'][0],
                                       run_globals['abs_hard_ROI'][1])
                                       
else:
    px_offsets['Cam_absorption']=(0,0)
    



#################################################################
# Pictures cropping
#################################################################

if run_globals['absorption_imaging']:
    abs_ROI_center = run_globals['abs_ROI_center']/effective_pixel_size['Cam_absorption']
    abs_ROI_size_x = run_globals['abs_ROI_size_x']/effective_pixel_size['Cam_absorption']
    abs_ROI_size_y = run_globals['abs_ROI_size_y']/effective_pixel_size['Cam_absorption']
    


    xslice = {
        'Cam_absorption' : slice(int(abs_ROI_center[0]-abs_ROI_size_x/2)-px_offsets['Cam_absorption'][0],
                           int(abs_ROI_center[0]+abs_ROI_size_x/2)-px_offsets['Cam_absorption'][0])
                
    } 
    yslice = {
        'Cam_absorption' : slice(int(abs_ROI_center[1]-abs_ROI_size_y/2)-px_offsets['Cam_absorption'][1],
                           int(abs_ROI_center[1]+abs_ROI_size_y/2)-px_offsets['Cam_absorption'][1])
    }

#################################################################
# Fitting and auxiliary functions
#################################################################
def cross_section(wlength, detuning, linewidth, I0, Isat):
    """
    Returns the absorption cross-section of a transition
    """
    sigma0 = 3 * wlength**2 / (2 * np.pi) #m2
    s = sigma0 / (1 + 4 * (detuning / linewidth)**2 + I0 / Isat) #m2
    return s

def oneD_Gaussian(x, x0, sigma, A, offset):#, B, x1):
    x = np.array(x)
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) + offset #+ B*(x-x1)**2
   
def bimodal(x, x0, sigma0, A0, offset, A1, x1, sigma1):#, B, xp):

    return A0*np.exp(-(x-x0)**2/(2*sigma0**2)) + A1*np.exp(-(x-x0)**2/(2*sigma1**2)) + offset #+ B*(x-xp)**2
   
#################################################################
# SCRIPT Starts here
#################################################################

sigma = cross_section(D2_wavelength, D2_detuning, D2_linewidth, I_in, D2_Isat)

OD = {}
OD_f={}
N_atoms = {}
warning = {}

def new_log(x) :
    log = np.where(x > 0., np.log(x), 0)
    log = np.nan_to_num(log)
    log_cap = np.where(np.abs(log) > OD_cap, 0, log)
    
    return log_cap
    
for cam in cam_names:
    #I = raw_imgs[cam]
    
    warning[cam] = ''
    
    # Extract the images 'before' and 'after' generated from camera.expose
    abs_atoms, abs_no_atoms, abs_dark = run.get_images(cam, 'MOT abs', 'atoms', 'no atoms', 'dark')

    abs_atoms = np.array(abs_atoms, dtype = 'float')
    abs_no_atoms = np.array(abs_no_atoms, dtype = 'float')
    abs_dark = np.array(abs_dark, dtype = 'float')
    
    
    
    
    
    abs_atoms = rotate(abs_atoms, angle[cam])
    abs_no_atoms = rotate(abs_no_atoms, angle[cam])
    abs_dark = rotate(abs_dark, angle[cam])    
    
    ys = yslice[cam]
    xs = xslice[cam]
    
    if abs_atoms[ys, xs].max() >  65e3 or abs_no_atoms[ys, xs].max() >  65e3:
        warning[cam] += 'saturated'

    
    # Compute the optical density, absorption cross-section and the number of atoms
    diff_atoms = np.subtract(abs_atoms[ys, xs], abs_dark[ys, xs])
    diff_light = np.subtract(abs_no_atoms[ys, xs], abs_dark[ys, xs])
    
    difftot = np.subtract(abs_no_atoms[ys, xs], abs_atoms[ys, xs])
    sumdiff = np.sum(difftot, axis = 0)
    
    
    if debug:
        plt.figure()
        plt.title("difference light - atoms")
        plt.imshow(difftot, cmap = plt.cm.bwr)
        plt.colorbar()
        plt.figure()
        plt.title("sum along some axis")
        plt.plot(sumdiff)
        plt.show()
    
    OD[cam] = np.nan_to_num(-new_log((diff_atoms)/(diff_light)))
    
    OD_f[cam]=np.nan_to_num(np.log((gaussian_filter(abs_no_atoms[ys, xs], sigma=2) - gaussian_filter(abs_dark[ys, xs], sigma=2))/( gaussian_filter(abs_atoms[ys, xs], sigma=2) - gaussian_filter(abs_dark[ys, xs], sigma=2) )))

    N_atoms[cam] = np.sum(OD[cam]) * pixel_area[cam] / sigma
    
    run.save_result(cam + '_Natoms', N_atoms[cam])
    run.save_result_array(cam + '_OD_image', OD[cam])
    run.save_result(cam + '_OD_peak', OD_f[cam].max())
    run.save_result(cam + '_warning', warning[cam])

#################################################################
# Debugging
#################################################################
    if debug:
        ROI_draw = patches.Rectangle(
            (xs.start,ys.start),
            OD[cam].shape[1],OD[cam].shape[0],
            linewidth=1,edgecolor='r',facecolor='none') 
        plt.figure('abs debug atoms'+cam)
        plt.imshow(abs_atoms)
        plt.gca().add_patch(ROI_draw)
        plt.colorbar()
        plt.show()

        plt.figure('abs debug no atoms'+cam)
        plt.imshow(abs_no_atoms)

        plt.colorbar()
        plt.show()

        plt.figure('abs debug dark'+cam)
        plt.imshow(abs_dark)

        plt.colorbar()
        plt.show()
        
        plt.figure('abs debug OD'+cam)
        plt.imshow(OD[cam],vmin = -0, vmax = None)
        plt.colorbar()
        plt.show()
        
        plt.figure('abs debug ODf'+cam)
        plt.imshow(OD_f[cam],vmin = -0, vmax = 0.3, cmap = 'inferno')
        plt.colorbar()
        plt.show()


#################################################################
# Fitting
#################################################################

def oneD_gaussian_fit(OD, region, axis):
    #calculate 1D integral
    from scipy.integrate import simps
    xf = np.arange(region[0], region[1])*effective_pixel_size[cam]
    oneDgauss = simps(OD, axis = axis, dx = effective_pixel_size[cam])
    
    # Estimate intial parameters
    oneDgauss_f = gaussian_filter(oneDgauss, sigma = 10)
    o0 = np.min(oneDgauss_f)
    oneDgauss_f -= o0
    integral = np.sum(oneDgauss_f) * effective_pixel_size[cam]
    x0 = xf[np.argmax(oneDgauss_f)]
    a0 = np.max(oneDgauss_f)
    
    w0 = integral/np.sqrt(2*np.pi)/a0 
    
    p0 = [x0, w0, a0, o0]
    

    
    try:
        fitparams, cov= curve_fit(oneD_Gaussian, xf, oneDgauss, p0 = p0)
        perr = np.sqrt(np.diag(cov))
    
        mu=ufloat(fitparams[0], perr[0])
        sigmafit=np.abs(ufloat(fitparams[1], perr[1]))
        amplitude=ufloat(fitparams[2], perr[2])
        offset=ufloat(fitparams[3], perr[3])
    except:
        print ('Fit not sucessfull')
        mu=ufloat('nan')
        sigmafit=ufloat('nan')
        amplitude=ufloat('nan')
        offset=ufloat('nan')
        fitparams = p0
        
        oneDgauss_f = gaussian_filter(oneDgauss, sigma = 10)
    
    fit_Natoms = np.sqrt(2*np.pi)*sigmafit*amplitude / sigma
    
    
    if tof[cam] != 0:
        T = m6Li / kB * (sigmafit**2)  / (tof[cam])**2 #* pixel_area[cam]
    else:
        T = ufloat(float('inf'), float('inf'))
    
    fitted_gauss = oneD_Gaussian(xf, *fitparams)
    
    
    if debug:
        plt.figure('fit'+str(axis))
        plt.plot(xf, oneDgauss, label = 'data')
        plt.plot(xf, fitted_gauss, label = 'fit')
        plt.plot(xf, oneD_Gaussian(xf,*p0), label = 'initial guess')
        
        plt.legend()
    
    return(oneDgauss, fit_Natoms, T, mu*1000, sigmafit*1000, fitted_gauss, xf*1000)


fit_OD=True
if fit_OD:
    axs = {'x': 0, 'y': 1}
    for cam in cam_names:
    
        ys = yslice[cam]
        xs = xslice[cam]
    
        xend = np.min([npx[cam][0], xs.stop])
        yend = np.min([npx[cam][1], ys.stop])

        fitresult = {}
        for key, axis in axs.items():
            region = [[xs.start+px_offsets[cam][0], xs.stop+px_offsets[cam][0]],[ys.start+px_offsets[cam][1], ys.stop+px_offsets[cam][1]]][axis]
            
            oneDgauss, fit_Natoms, T, mu, sigmafit, fitted_gauss, xf=oneD_gaussian_fit(OD[cam],region, axis=axis)
            
            run.save_results(cam+"_N"+key, fit_Natoms.n,cam+"_N"+key+'_err', 
                            fit_Natoms.std_dev, cam+"_c"+key, mu.n,cam+"_c"+key+'_err', mu.std_dev, 
                            cam+"_s"+key, sigmafit.n, cam+"_s"+key+'_err', sigmafit.std_dev, cam+'_axes'+key, cam_axes[cam][key])
            run.save_result_array(cam+'_'+key+'fit', fitted_gauss)
            run.save_result_array(cam+'_'+key+'sum', oneDgauss)
            run.save_result_array(cam+'_'+key+'grid', xf)
                            
            fitresult[key]={}
            fitresult[key]['oneDgauss']=oneDgauss
            fitresult[key]['fitted_gauss']=fitted_gauss
            fitresult[key]['fit_Natoms']=fit_Natoms
            fitresult[key]['T']=T
            fitresult[key]['mu']=mu
            fitresult[key]['sigmafit']=sigmafit
            fitresult[key]['xf']=xf
            
            if mu == ufloat('nan'):
                warning[cam] = ' '+key+'fit failed'

#################################################################
# Plot results
#################################################################
        if use_matplotlib:
            plt.figure(f'AI {cam}')
            gs = gridspec.GridSpec(2, 3, height_ratios = (2, 1), width_ratios = (11, 20, 1))
            
            # Projection along X direction
            yplot = plt.subplot(gs[0,0])
            ysum = fitresult['y']['oneDgauss']
            ygrid = fitresult['y']['xf']
            plt.plot(ysum, ygrid, '.', label = 'data')
            plt.plot(fitresult['y']['fitted_gauss'], ygrid, linewidth = 2)
            plt.ylim(ygrid[-1], ygrid[0])
            plt.xlabel('OD (a.u.)')
            plt.ylabel('y (mm)')
            plt.legend()
            plt.grid()
            yplot.axes.xaxis.set_label_position('top')
            yplot.axes.xaxis.set_tick_params(labeltop = False, labelbottom = False,top=True) 
            
            
            
            # Projection along Y direction
            xplot = plt.subplot(gs[1,1])
            xsum = fitresult['x']['oneDgauss']
            xgrid = fitresult['x']['xf']
            plt.plot(xgrid, xsum, '.', label = 'data')
            plt.plot(xgrid, fitresult['x']['fitted_gauss'], linewidth = 2)
            plt.xlim(xgrid[0], xgrid[-1])
            plt.xlabel('x (mm)')
            plt.ylabel('OD (a.u.)')
            plt.legend()
            plt.grid()
            xplot.axes.yaxis.set_label_position('right')
            # xplot.yticks(xgrid*effective_pixel_size[cam]*1000)
            xplot.axes.yaxis.set_tick_params(labelright = False, labelleft = False, right=True)
            
            
            # 2D OD image
            ODimg=plt.subplot(gs[0,1], sharex = xplot, sharey = yplot)
            plt.imshow(OD[cam],extent = [xgrid[0], xgrid[-1],ygrid[-1], ygrid[0]],
                        vmin=0.
                        ,vmax=0.2
                        )#plt.get_cmap('bone'))
            #ODimg.axes.xaxis.set_ticklabels([])
            #ODimg.axes.yaxis.set_ticklabels([])
            plt.plot(fitresult['x']['mu'].n,fitresult['y']['mu'].n, 'r+')
            plt.title(cam + ' OD ' + 'ToF '+ str(1000*tof[cam]) + 'ms (OD cap at '+str(OD_cap)+')\n')
            plt.subplot(gs[0,2])
            plt.axis('off')
            plt.colorbar(fraction = 2)
            
            try:
                plt.figtext(0.6, 0.93, warning[cam], color = 'red')
            except:
                plt.figtext(0.6, 0.93, 'all good!', color = 'green')
            
            # Fit results
            plt.subplot(gs[1,0])
            colLabels = ['x', 'y']
            rowLabels = ['N ('+str(np.round(N_atoms[cam],0))+')', 'T (K)', 'c (mm)', 's (mm)']
            cellText = [[fitresult["x"]["fit_Natoms"], str(fitresult['y']['fit_Natoms'])], 
                        [fitresult['x']['T'], fitresult['y']['T']], 
                        [fitresult['x']['mu'],fitresult['y']['mu']], 
                        [fitresult['x']['sigmafit'], fitresult['y']['sigmafit']]]
            result_table=plt.table(cellText = cellText, rowLabels = rowLabels, colLabels = colLabels, loc = 'center', bbox = [-0.1, -0.3, 1.25, 1.5], cellLoc='center')
            result_table.auto_set_font_size(False)
            result_table.set_fontsize(7.)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
