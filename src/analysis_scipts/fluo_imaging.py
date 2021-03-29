
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
from scipy.ndimage import rotate
from matplotlib import gridspec
from pathlib import Path
from scipy.ndimage import gaussian_filter
from uncertainties import ufloat
from uncertainties.umath import *  

import matplotlib.patches as patches


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
df = lyse.data(h5_path)


debug=False#run_globals['debug_abs_imaging']
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

blue_wavelength = 460.42e-9
blue_linewidth = 19.35e6
blue_Isat = D2_Isat*(D2_wavelength/blue_wavelength)**3 * blue_linewidth / D2_linewidth



#################################################################
# Experimental parameters
#################################################################

# Get a dictionary of the global variables used in this shot
run_globals = run.get_globals()

cam_names = []
if run_globals['MOT_fluo'] :
    cam_names.append('Cam_fluorescence')
    cam_names.append('Cam_fluorescence_side')
if run_globals['Blue_fluo'] :
    cam_names.append('PCOedge')


run.save_result('cams', cam_names)

Gamma = {'Cam_fluorescence':D2_linewidth,
         'Cam_fluorescence_side':D2_linewidth,
         'PCOedge': blue_linewidth}
         
excited_state_mean_occupation = {'Cam_fluorescence': 0.5,
                                 'Cam_fluorescence_side': 0.5,
                                 'PCOedge':1/3.}

cam_axes = {
    'Cam_fluorescence' : {'x' : 'y' , 'y' : r'x'} ,
    'Cam_fluorescence_side' : {'x' : '-z' , 'y' : '-x+y'} ,
    'PCOedge' : {'x' : 'y' , 'y' : 'z'}
    }

pixel_size = {
    'Cam_fluorescence' : run_globals['Flir_pixel_size'],
    'Cam_fluorescence_side' : run_globals['Flir_pixel_size'],
    'PCOedge' : run_globals['PCO_pixel_size']
}

exposure = {
    'Cam_fluorescence' : run_globals['MOT_fluo_exposure'][0],
    'Cam_fluorescence_side' : run_globals['MOT_fluo_exposure'][1],
    'PCOedge' : run_globals['bluefluo_exposure']
}

magnification = {
    'Cam_fluorescence' : run_globals['redfluo_magnification'],
    'Cam_fluorescence_side' :run_globals['redfluo_side_magnification'],
    'PCOedge' : run_globals['bluefluo_magnification']
}

#numerical aperature of collection lens
NA = {
    'Cam_fluorescence' : run_globals['redfluo_NA'],
    'Cam_fluorescence_side' : run_globals['redfluo_side_NA'],
    'PCOedge' : run_globals['PCO_NA']
}

#Quantum efficiency
QE = {
    'Cam_fluorescence' : run_globals['Flir_QE'],
    'Cam_fluorescence_side' : run_globals['Flir_QE'],
    'PCOedge' : run_globals['PCO_QE']
}

#Quantum efficiency
I_Isat = {
    'Cam_fluorescence' : run_globals['redfluo_I_Isat'],
    'Cam_fluorescence_side' : run_globals['redfluo_I_Isat'],
    'PCOedge' : run_globals['bluefluo_I_Isat']
}

npx = {
    'Cam_fluorescence' : run_globals['Flir_npx'],
    'Cam_fluorescence_side' : run_globals['Flir_npx'],
    'PCOedge' : run_globals['PCO_npx']
}

px_offsets = {}

if run_globals['redfluo_crop_hardware']:
    px_offsets['Cam_fluorescence'] =  (run_globals['redfluo_hard_ROI'][0],
                                       run_globals['redfluo_hard_ROI'][1])
                                       
else:
    px_offsets['Cam_fluorescence']=(0,0)
    
if run_globals['redfluo_side_crop_hardware']:
    px_offsets['Cam_fluorescence_side'] =  (run_globals['redfluo_side_hard_ROI'][0],
                                            run_globals['redfluo_side_hard_ROI'][1])                                      
else:
    px_offsets['Cam_fluorescence_side']=(0,0)

if run_globals['bluefluo_crop_hardware']:
    px_offsets['PCOedge'] =  (run_globals['bluefluo_hard_ROI'][0],
                                            run_globals['bluefluo_hard_ROI'][1])
                                       
else:
    px_offsets['PCOedge']=(0,0)
    
effective_pixel_size = {
    name : pixel_size[name] * magnification[name]
    for name in cam_names
}

pixel_area = {
    name : (effective_pixel_size[name])**2
    for name in cam_names
}

DT_center = {
    'Cam_fluorescence': [0.25, -0.1],
    'Cam_fluorescence_side': [0, 0], #this is not calibrated yet!
    'PCOedge': [0, 0] #this is not calibrated yet!
}
#################################################################
# Pictures cropping
#################################################################
if run_globals['MOT_fluo'] :
    redfluo_ROI_center = run_globals['redfluo_ROI_center']/effective_pixel_size['Cam_fluorescence']
    redfluo_ROI_size = run_globals['redfluo_ROI_size']/effective_pixel_size['Cam_fluorescence']
    redfluo_side_ROI_center = run_globals['redfluo_side_ROI_center']/effective_pixel_size['Cam_fluorescence_side']
    redfluo_side_ROI_size = run_globals['redfluo_side_ROI_size']/effective_pixel_size['Cam_fluorescence_side']

if run_globals['Blue_fluo'] :
    bluefluo_ROI_center = run_globals['bluefluo_ROI_center']/effective_pixel_size['PCOedge']
    bluefluo_ROI_size = run_globals['bluefluo_ROI_size']/effective_pixel_size['PCOedge']

corners = {
    'Cam_fluorescence': np.array([run_globals['redfluo_corner_tr'], 
                                run_globals['redfluo_corner_tl'],
                                run_globals['redfluo_corner_br'],
                                run_globals['redfluo_corner_bl']]),
                                 
    'Cam_fluorescence_side': np.array([run_globals['redfluo_side_corner_tr'], 
                                run_globals['redfluo_side_corner_tl'],
                                run_globals['redfluo_side_corner_br'],
                                run_globals['redfluo_side_corner_bl']]),
    }

def intersection(corners,cam) :
    x1=corners[cam][1][0] ; y1=corners[cam][1][1]
    x2=corners[cam][0][0] ; y2=corners[cam][0][1]
    x3=corners[cam][3][0] ; y3 = corners[cam][3][1]
    x4=corners[cam][2][0] ; y4 = corners[cam][2][1]
    D = (y4-y1)/(x4-x1) - (y3-y2)/(x3-x2)
    N = x1*(y4-y1)/(x4-x1) - x2*(y3-y2)/(x3-x2) +y2-y1
    xA = N/D
    yA = y1 + (xA-x1)*(y4-y1)/(x4-x1)
    return (int(xA),int(yA))
    

centerpix = {
    cam : intersection(corners,cam)
    for cam in ['Cam_fluorescence','Cam_fluorescence_side']
    }
if run_globals['Blue_fluo'] :
    centerpix['PCOedge'] = bluefluo_ROI_center
if run_globals['MOT_fluo'] :
    ROI_soft = {
        'Cam_fluorescence' : {'center' : redfluo_ROI_center, 'size' : redfluo_ROI_size},
        'Cam_fluorescence_side' : {'center' : redfluo_side_ROI_center , 'size' : redfluo_side_ROI_size}
        }
if run_globals['Blue_fluo'] :
    ROI_soft['PCOedge'] = {'center' : centerpix['PCOedge'] , 'size' : bluefluo_ROI_size}

    
xslice = { cam : slice(
            int(ROI_soft[cam]['center'][0]-ROI_soft[cam]['size']/2.-px_offsets[cam][0]),
            int(ROI_soft[cam]['center'][0]+ROI_soft[cam]['size']/2.-px_offsets[cam][0])
            )
            for cam in cam_names
        }
# print(xslice)
yslice = { cam : slice(
            int(ROI_soft[cam]['center'][1]-ROI_soft[cam]['size']/2.-px_offsets[cam][1]),
            int(ROI_soft[cam]['center'][1]+ROI_soft[cam]['size']/2.-px_offsets[cam][1])
            )
            for cam in cam_names
        }

#################################################################
# Fitting and auxiliary functions
#################################################################

def oneD_Gaussian(x, x0, sigma, A, offset):#, B, x1):
    x = np.array(x)
    return A*np.exp(-(x-x0)**2/(2*sigma**2)) + offset #+ B*(x-x1)**2
   
def bimodal(x, x0, sigma0, A0, offset, A1, x1, sigma1):#, B, xp):

    return A0*np.exp(-(x-x0)**2/(2*sigma0**2)) + A1*np.exp(-(x-x0)**2/(2*sigma1**2)) + offset #+ B*(x-xp)**2
   
#################################################################
# SCRIPT Starts here
#################################################################



# Save this very routine in the sequence's h5 file
#lqg.save_routine(__file__, path)

#################################################################
# Load images and compute OD and columns density
#################################################################
# tof = singleshot_data['TOF_duration']

diff = {}
diff_f={}
diff_bg = {}
N_atoms = {}
Np = {}
warning = {}


# cam_names = ['Cam_fluorescence']


for i,cam in enumerate(cam_names):
    #I = raw_imgs[cam]
    
    warning[cam] = ''
    
    print (f'Anaysis of data of camera {cam}')
    
    # Extract the images 'before' and 'after' generated from camera.expose
    try:
        fluo_atoms, fluo_no_atoms,fluo_no_atoms2 = run.get_images(cam, 'MOT fluo', 'atoms', 'no atoms' , 'no atoms2')
    except:
        del cam_names[i]
        continue
    fluo_atoms = np.array(fluo_atoms, dtype = 'float')
    fluo_no_atoms = np.array(fluo_no_atoms, dtype = 'float')
    fluo_no_atoms2 = np.array(fluo_no_atoms2, dtype = 'float')
    
    ys = yslice[cam]
    xs = xslice[cam]
    
    #check for saturation in ROI
    if fluo_atoms[ys, xs].max() >  65e3 or fluo_no_atoms[ys, xs].max() >  65e3:
        warning[cam] += 'saturated'

    
    
    # Compute difference of images with and without atoms
    diff[cam] = fluo_atoms[ys, xs] - fluo_no_atoms[ys, xs]
    
    diff_bg[cam] = fluo_no_atoms[ys, xs] - fluo_no_atoms2[ys, xs]
    
    diff_f[cam]=gaussian_filter(diff[cam], sigma=5)
    
    #crop image around MOT
    diff_f[cam]=gaussian_filter(diff[cam], sigma=5)
    # print(cam)
    #AMax=np.amax(diff_f[cam])
    #max_coord=np.where(diff_f[cam]==np.amax(diff_f[cam]))
    
    if debug:
        ROI_draw = patches.Rectangle(
            (xs.start+px_offsets[cam][0],ys.start+px_offsets[cam][1]),
            diff[cam].shape[1],diff[cam].shape[0],
            linewidth=1,edgecolor='r',facecolor='none')    
        plt.figure('fluo debug atoms '+cam)
        plt.imshow(fluo_atoms,
                    extent = [px_offsets[cam][0]-0.5, fluo_atoms.shape[1]+px_offsets[cam][0]-0.5,fluo_atoms.shape[0]+px_offsets[cam][1]-0.5,px_offsets[cam][1]-0.5]
                    )
        plt.gca().add_patch(ROI_draw)
        
        if cam in corners:
            plt.plot(np.array(corners[cam])[0::3][:,0], np.array(corners[cam])[0::3][:,1],'b+-')
            plt.plot(np.array(corners[cam])[1:3][:,0], np.array(corners[cam])[1:3][:,1],'b+-')
        
        
        plt.show()
    
        ROI_draw = patches.Rectangle(
            (xs.start,ys.start),
            diff[cam].shape[1],diff[cam].shape[0],
            linewidth=1,edgecolor='r',facecolor='none')
        plt.figure('fluo debug no atoms '+cam)
        plt.imshow(fluo_no_atoms)
        plt.gca().add_patch(ROI_draw)
        plt.show()
        
    
        ROI_draw = patches.Rectangle(
            (xs.start,ys.start),
            diff[cam].shape[1],diff[cam].shape[0],
            linewidth=1,edgecolor='r',facecolor='none')
        plt.figure('fluo debug no atoms2 '+cam)
        plt.imshow(fluo_no_atoms2)
        plt.gca().add_patch(ROI_draw)
        plt.show()
    
        plt.figure('fluo debug diff '+cam)
        plt.imshow(diff[cam])
        plt.title(f'{np.sum(diff[cam]):.2e}')
        #plt.plot(max_coord[1], max_coord[0], 'b+', label = 'Detected MOT fluo')
        plt.plot(ROI_soft[cam]['size']/2,ROI_soft[cam]['size']/2, 'r+',markersize=10
        , label = 'Cavity center')
        plt.colorbar()
        plt.legend()
        plt.show()
        
        plt.figure('fluo debug diff histogram'+cam)
        # print(plt.hist(diff[cam].flatten(),21,(-10.5,10.5)))
        plt.show()
    
    #crop around MOT
    #x0=(np.int(max_coord[0][0]))-15
    #x1=(np.int(max_coord[0][0]))+15
    #y0=(np.int(max_coord[1][0]))-15
    #y1=(np.int(max_coord[1][0]))+15
    
    #diff[cam] = diff[cam][x0:x1,y0:y1]
    #diff_f[cam] = diff_f[cam][x0:x1,y0:y1]
    
    
    
    #electron count of shot
    fluo=np.sum(diff[cam])
    #estimated background noice
    fluo_e = np.std(diff_bg[cam]) *np.sqrt(diff_bg[cam].shape[0] *  diff_bg[cam].shape[1])
    
    fluo = ufloat(fluo, fluo_e)
    
    delta = Gamma[cam]/2

    s = I_Isat[cam] * 1/((2*delta/Gamma[cam])**2 + 1)
    gamma = excited_state_mean_occupation[cam] * s/(s+1) * (Gamma[cam])

    #solid angle of objective
    Omega = 2*np.pi*(1 - np.sqrt(1 - NA[cam]**2))

    #electron rate of single atom
    gamma_m = QE[cam] * (Omega/4*np.pi) * gamma 
    
    # print (QE[cam] * (Omega/4*np.pi))
    # print (gamma)
    #number of electrons per atom
    Np[cam] = gamma_m * exposure[cam]
    
    # print (f'Number of counts per atoms per second:   {gamma_m}', (Omega/4*np.pi))

    N_atoms[cam] = fluo/Np[cam]
    run.save_results(cam + '_bkgd_counts', np.sum(fluo_no_atoms),cam + '_bkgd_Natoms', np.sum(diff_bg[cam])/Np[cam])
    run.save_result(cam + '_Natoms', N_atoms[cam].n)
    run.save_result(cam + '_Natoms_err', N_atoms[cam].std_dev)
    run.save_result_array(cam + '_diff_image', diff[cam])
    run.save_result(cam + '_warning', warning[cam])

#################################################################
# Fitting
#################################################################

def oneD_gaussian_fit(diff, region, axis):
    
    
    #calculate 1D integral
    from scipy.integrate import simps
    xf = np.arange(region[0], region[1])*effective_pixel_size[cam]*1000
    oneDgauss = simps(diff, axis = axis, dx = 1/(effective_pixel_size[cam]*1000))
    
    # Estimate intial parameters
    oneDgauss_f = gaussian_filter(oneDgauss, sigma = 10)
    o0 = np.min(oneDgauss_f)
    oneDgauss_f -= o0
    integral = np.sum(oneDgauss_f) * effective_pixel_size[cam]*1000 
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

    
    fit_Natoms = np.sqrt(2*np.pi)* sigmafit * amplitude / Np[cam]
    fitted_gauss = oneD_Gaussian(xf, *fitparams)

    return(oneDgauss, fit_Natoms, mu, sigmafit, fitted_gauss, xf)

fit_diff=True
if fit_diff:
    axs = {'x': 0, 'y': 1}
    for cam in cam_names:
        ys = yslice[cam]
        xs = xslice[cam]
        
        fitresult = {}
        for key, axis in axs.items():
            region = [
                    [xs.start+px_offsets[cam][0]-ROI_soft[cam]['center'][0], xs.stop+px_offsets[cam][0]-ROI_soft[cam]['center'][0]],
                    [ys.start+px_offsets[cam][1]-ROI_soft[cam]['center'][1], ys.stop+px_offsets[cam][1]-ROI_soft[cam]['center'][1]]
                    ][axis]

            # print(np.shape(diff[cam]))
            # print(region[1] - region[0])
            oneDgauss, fit_Natoms, mu, sigmafit, fitted_gauss, xf =oneD_gaussian_fit(diff[cam],region, axis=axis)
            
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
            fitresult[key]['mu']=mu
            fitresult[key]['sigmafit']=sigmafit
            fitresult[key]['xf']=xf
            
            if mu == ufloat('nan'):
                warning[cam] = ' '+key+'fit failed'


#################################################################
# Plot results
#################################################################

        xsum = fitresult['x']['oneDgauss']
        xgrid = fitresult['x']['xf']

        ysum = fitresult['y']['oneDgauss']
        ygrid = fitresult['y']['xf']
        if use_matplotlib:
            plt.figure(f'FI {cam}')
            gs = gridspec.GridSpec(2, 3, height_ratios = (2, 1), width_ratios = (11, 20, 1))
            
            # Integrate along X direction, display along y
            yplot = plt.subplot(gs[0,0])

            plt.plot(ysum, ygrid, '.', label = 'data')
            plt.plot(fitresult['y']['fitted_gauss'], ygrid, linewidth = 2)
            plt.ylim(ygrid[-1], ygrid[0])
            plt.xlabel(r'diff (kcounts)')
            plt.ylabel(cam_axes[cam]['y']+' (mm)')
            plt.legend()
            plt.grid()
            yplot.axes.xaxis.set_label_position('top')
            yplot.axes.xaxis.set_tick_params(labeltop = True, labelbottom = False,top=True)
            yplot.axes.set_xticks(yplot.axes.get_xticks().tolist())
            yplot.axes.set_xticklabels(["%d" %(diff_val/1e3) for diff_val in yplot.axes.get_xticks().tolist()])
            
            
            
            # Integrate along Y direction, display along X
            xplot = plt.subplot(gs[1,1])
            
            plt.plot(xgrid, xsum, '.', label = 'data')
            plt.plot(xgrid, fitresult['x']['fitted_gauss'], linewidth = 2)
            plt.xlim(xgrid[0], xgrid[-1])
            plt.xlabel(cam_axes[cam]['x']+' (mm)')
            plt.ylabel('diff (kcounts)')
            plt.grid()
            xplot.axes.yaxis.set_label_position('right')
            # xplot.yticks(xgrid*effective_pixel_size[cam]*1000)
            xplot.axes.yaxis.set_tick_params(labelright = True, labelleft = False, right=True)
            xplot.axes.set_yticks(xplot.axes.get_yticks().tolist())
            xplot.axes.set_yticklabels(["%d" %(diff_val/1e3) for diff_val in xplot.axes.get_yticks().tolist()])
            # xplot.axes.yaxis.set_ticklabels([])
            
            # 2D diff image
            diffimg=plt.subplot(gs[0,1], sharex = xplot, sharey = yplot)
            plt.imshow(diff[cam],extent = [xgrid[0], xgrid[-1],ygrid[-1], ygrid[0]]
                        ,vmin=-500
                        ,vmax=None
                        )
            #plt.get_cmap('bone'))
            #diffimg.axes.xaxis.set_ticklabels([])
            #diffimg.axes.yaxis.set_ticklabels([])
            plt.plot(fitresult['x']['mu'].n,fitresult['y']['mu'].n, 'b+',label='MOT center')
            plt.plot(0,0, 'r+',markersize=10, label = 'Cavity center')
            if cam=='Cam_fluorescence':
                plt.plot(DT_center[cam][0],DT_center[cam][1], 'g+',markersize=10, label = 'DT center')
            #plt.plot(ROI_soft[cam]['center'][0]*effective_pixel_size[cam]*1000,ROI_soft[cam]['center'][1]*effective_pixel_size[cam]*1000, 'r+',markersize=10, label = 'Cavity center')
            plt.legend()
            plt.title(cam + ' diff ')
            plt.subplot(gs[0,2])
            plt.axis('off')
            plt.colorbar(fraction = 2)
            
            #print warning
            try:
                plt.figtext(0.6, 0.95, warning[cam], color = 'red')
            except:
                plt.figtext(0.6, 0.95, 'all good!', color = 'green')
            # Fit results
            plt.subplot(gs[1,0])
            colLabels = [cam_axes[cam]['x'], cam_axes[cam]['y']]
            rowLabels = ['N ('+'{:.1ueS}'.format(N_atoms[cam])+')', 'c (mm)', 's (mm)']
            cellText = [['{:.1ueS}'.format(fitresult["x"]["fit_Natoms"]), 
                            '{:.1ueS}'.format(fitresult['y']['fit_Natoms'])], ['{:.1uS}'.format(fitresult['x']['mu']),
                            '{:.1uS}'.format(fitresult['y']['mu'])], ['{:.1uS}'.format(fitresult['x']['sigmafit']),
                            '{:.1uS}'.format(fitresult['y']['sigmafit'])]]
            result_table=plt.table(cellText = cellText, rowLabels = rowLabels, colLabels = colLabels, loc = 'center', bbox = [-0.1, -0.3, 1.2, 1.2], cellLoc='center')
            result_table.auto_set_font_size(False)
            result_table.set_fontsize(7.)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        
        
