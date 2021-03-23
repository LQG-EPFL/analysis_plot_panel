# Installation

## Download

The Analysis Plot Panel is though to be used as a single shot routine in lyse. In order to set this up, one first has to clone the git into the folder where all the analysis scripts are stored (usually: `C:\Users\user_name\labscript-suite\userlib\analysislib\experiment_name`), where `user_name` and `experiment_name` have to be changed individually.

## Additional packages required

The analysis plot panel requires the following packages:

- `lyse`: Installed with the labscript suite 

- `pyqt5`: Installed with the labscript suite 

- `pyqtgraph`: Installed with the labscript suite 

- `h5py`: Installed with the labscript suite

- `sortedcontainers`: pip install sortedcontainers

!!! warning 
	All these packages should be installed in the correct python environment. Check the [installation guide of labscript](https://docs.labscriptsuite.org/en/stable/installation/setting-up-an-environment/#choosing-an-installation-method) for more details.

## Setup of lyse

As a last step, the `analysis_plot_panel_lyse_routine.py` file should be loaded as a single shot routine in lyse:

![](lyse_config.PNG)

