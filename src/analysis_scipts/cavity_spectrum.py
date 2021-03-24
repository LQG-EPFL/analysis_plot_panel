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
import time

#################################################################
# Load h5
#################################################################

import cProfile
import pstats

profile = cProfile.Profile()
profile.enable()

if lyse.spinning_top:
    # If so, use the filepath of the current shot
    h5_path = lyse.path
else:
    # If not, get the filepath of the last shot of the lyse DataFrame
    h5_path = lyse.h5_paths().iloc[-1]

# Instantiate a lyse.Run object for this shot
run = lyse.Run(h5_path)

single_plot = False
fit = True
hist = True
multi_plot = False
# Get a dictionary of the global variables used in this shot
run_globals = run.get_globals()

device_name = 'TH260_nano'

freq_bin_0 = .5 #MHz

total_results_dict = {'single':{}, 'array':{}}

freqs_all = {}
nphotons_all = {}
omekappa_all = {}

def analyse_sweep(name, frametype, f0, f1, duration):
    warning = ''
    
    
    
    time_array = run.get_time_arrays(device_name, 'cavity spectrum '+name, frametype)[0]
    
    t2freq = lambda t: t/duration * (f1 - f0) + f0 
    freq2t = lambda freq: duration * (freq - f0)/(f1 - f0)
    
    if single_plot:
        fig = plt.figure('spectrum of '+name+' '+frametype )
        ax1 = fig.add_subplot(111)
    
    n_bins = int(abs((f1 - f0)/freq_bin_0))
    if n_bins < 100 : 
        n_bins = 100
        freq_bin = abs(f1-f0)/n_bins
    else :
        freq_bin = freq_bin_0
    
    if hist:
        t = time.time()
        nphotons, t_bins = np.histogram(time_array, bins = n_bins, range = (0, duration))
    else:
        nphotons, t_bins = np.histogram([0], bins = n_bins, range = (0, duration))
    freq_bins = t2freq(t_bins)
    
    ts = (t_bins[:-1] + t_bins[1:])/2
    freqs = t2freq(ts)
    
    freqs_all[name + ' ' + frametype ] = freqs
    nphotons_all[name + ' ' + frametype ] = nphotons
    
    if single_plot:
        plt.bar(freqs, nphotons, width = freq_bin)
    
    
    
    cnt2rate = lambda c: c/(t_bins[1]-t_bins[0])*1e-6
    rate2cnt = lambda r: r*(t_bins[1]-t_bins[0])/1e-6
    # print(t_bins)
    
    if single_plot:
        ax2 = ax1.secondary_xaxis("top", functions=(freq2t,t2freq))
        ax3 = ax1.secondary_yaxis("right", functions = (cnt2rate,rate2cnt))

        ax2.set_xlabel("time [s]")
        ax1.set_xlabel("freq [MHz]")
        ax3.set_ylabel(r"counts per $\mu$s")

        plt.ylabel('photon counts')
        plt.xlim(f0,f1)
        plt.ylim(0,None)
        plt.hlines(rate2cnt(10),freq_bins[0],freq_bins[-1],linestyles= '--')
        
    if fit:
        if len(time_array) > 0:
            from scipy.optimize import curve_fit
            try:
                lorenzian = lambda omega, omega0, kappa, A, offset: A * (kappa/2)**2 / ((omega - omega0)**2 + (kappa/2)**2) + offset

                
                para, cov = curve_fit(lorenzian, 2*np.pi*freqs, nphotons, p0 = [2*np.pi*freqs[np.argmax(nphotons)], 2*np.pi*0.5, nphotons.max(), 0])

                freqsp = np.linspace(f0, f1, 100)
                
                if single_plot:
                    ax1.plot(freqsp, lorenzian(2*np.pi*freqsp, *para), 'C1', label = 'lorenzian fit')
                
                kappa = np.abs(para[1])
                omega0 = para[0]
                A= para[2]
                offset = para[3]
                omekappa_all[name + ' ' + frametype ] = para
                
            except:
                import traceback
                traceback.print_exc()
                
                kappa = np.nan
                omega0 = np.nan
                A = np.nan
                offset = np.nan
                
                warning += ' fit failed'
        else:
            kappa = np.nan
            omega0 = np.nan
            A = np.nan
            offset = np.nan
            
            warning += ' no photons in shot'
            
    else:
        freq_array = t2freq(time_array)
        kappa = np.std(freq_array)*2*np.pi
        omega0 = np.mean(freq_array)*2*np.pi
        
        A = 0
        offset = np.max(nphotons)
    
    if single_plot:
        plt.vlines(omega0/2/np.pi,ymin=0, ymax=nphotons.max(),color='orange')
        plt.title(name+' '+frametype+'  '
                + r' $\kappa = 2 \pi \times $'+f'{(kappa/2/np.pi):.3f}' + ' MHz \t' 
                + r' $\omega_0 = 2 \pi \times $'+f'{omega0/2/np.pi:.3f}' + ' MHz' )
    
    #plt.legend()

    total_results_dict['single'] = {**total_results_dict['single'],
                                    **{name+' '+frametype+' kappa': kappa,
                                       name+' '+frametype+' omega0': omega0,
                                       name+' '+frametype+' A': A,
                                       name+' '+frametype+' offset': offset,
                                       name+' '+frametype+' n_photons_total': len(time_array),
                                       name+' '+frametype+' warning': warning,
                                       name+' '+frametype+' f0': f0,
                                       name+' '+frametype+' f1': f1,
                                       name+' '+frametype+' duration': duration}}
    
    total_results_dict['array'] = {**total_results_dict['array'],
                                    **{name+' '+frametype+' n_photons': nphotons, 
                                       name+' '+frametype+' ts': ts,
                                       name+' '+frametype+' freqs': freqs}}
    
    if single_plot:
        plt.tight_layout()
    
    return freqs, nphotons

for spec_counter in range(len(run_globals['cavity_probe_names'])):

    f0 = run_globals['cavity_probe_detunings'][spec_counter] - run_globals['cavity_probe_sweep_ranges'][spec_counter]/2
    f1 = run_globals['cavity_probe_detunings'][spec_counter] + run_globals['cavity_probe_sweep_ranges'][spec_counter]/2
    analyse_sweep(run_globals['cavity_probe_names'][spec_counter],
                  run_globals['cavity_probe_frametypes'][spec_counter], 
                  f0, f1, 
                  run_globals['cavity_probe_sweep_durations'][spec_counter])

lorenzian = lambda omega, omega0, kappa, A, offset: A * (kappa/2)**2 / ((omega - omega0)**2 + (kappa/2)**2) + offset

if multi_plot:
    plt.figure('all spectra')
    for key in freqs_all.keys() :
        if key != 'sidepump atoms' :
            plt.bar(freqs_all[key],nphotons_all[key], label=key)
            if fit : 
                plt.plot(freqs_all[key], lorenzian(2*np.pi*freqs_all[key], *omekappa_all[key]),'--',color='black')
    plt.title('all spectra')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('photon counts')
    plt.legend()
    plt.xlim(-60,10)
    plt.show()

run.save_total_results_dict(total_results_dict)    

profile.disable()
ps = pstats.Stats(profile)
ps.sort_stats('cumtime')
ps.print_stats(10)