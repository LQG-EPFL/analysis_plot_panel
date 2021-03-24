#####################################################################
#                                                                   #
# /__init__.py                                                      #
#                                                                   #
# Copyright 2013, Monash University                                 #
#                                                                   #
# This file is part of the program lyse, in the labscript suite     #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################

from lyse.dataframe_utilities import get_series_from_shot as _get_singleshot
from labscript_utils.dict_diff import dict_diff
import os
import socket
import pickle as pickle
import inspect
import sys
import threading

import labscript_utils.h5_lock
import h5py
from labscript_utils.labconfig import LabConfig
import pandas
from numpy import array, ndarray
import types

from .__version__ import __version__

from labscript_utils import dedent
from labscript_utils.ls_zprocess import zmq_get

from labscript_utils.properties import get_attributes, get_attribute, set_attributes
LYSE_DIR = os.path.dirname(os.path.realpath(__file__))

# If running stand-alone, and not from within lyse, the below two variables
# will be as follows. Otherwise lyse will override them with spinning_top =
# True and path <name of hdf5 file being analysed>:
spinning_top = False
# data to be sent back to the lyse GUI if running within lyse
_updated_data = {}
# dictionary of plot id's to classes to use for Plot object
_plot_classes = {}
# A fake Plot object to subclass if we are not running in the GUI
Plot=object
# An empty dictionary of plots (overwritten by the analysis worker if running within lyse)
plots = {}
# A threading.Event to delay the 
delay_event = threading.Event()
# a flag to determine whether we should wait for the delay event
_delay_flag = False

# get port that lyse is using for communication
try:
    _labconfig = LabConfig(required_params={"ports": ["lyse"]})
    _lyse_port = int(_labconfig.get('ports', 'lyse'))
except Exception:
    _lyse_port = 42519

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = None


class _RoutineStorage(object):
    """An empty object that analysis routines can store data in. It will
    persist from one run of an analysis routine to the next when the routine
    is being run from within lyse. No attempt is made to store data to disk,
    so if the routine is run multiple times from the command line instead of
    from lyse, or the lyse analysis subprocess is restarted, data will not be
    retained. An alternate method should be used to store data if desired in
    these cases."""

routine_storage = _RoutineStorage()


def data(filepath=None, host='localhost', port=_lyse_port, timeout=5):
    if filepath is not None:
        return _get_singleshot(filepath)
    else:
        df = zmq_get(port, host, 'get dataframe', timeout)
        try:
            padding = ('',)*(df.columns.nlevels - 1)
            try:
                integer_indexing = _labconfig.getboolean('lyse', 'integer_indexing')
            except (LabConfig.NoOptionError, LabConfig.NoSectionError):
                integer_indexing = False
            if integer_indexing:
                df.set_index(['sequence_index', 'run number', 'run repeat'], inplace=True, drop=False)
            else:
                df.set_index([('sequence',) + padding,('run time',) + padding], inplace=True, drop=False)
                df.index.names = ['sequence', 'run time']
        except KeyError:
            # Empty DataFrame or index column not found, so fall back to RangeIndex instead
            pass
        df.sort_index(inplace=True)
        return df
        
def h5_paths(filepath=None, host='localhost', port=_lyse_port, timeout=5):
    if filepath is not None:
        return _get_singleshot(filepath)
    else:
        df = zmq_get(port, host, 'get h5_paths', timeout)
        # try:
            # padding = ('',)*(df.columns.nlevels - 1)
            # try:
                # integer_indexing = _labconfig.getboolean('lyse', 'integer_indexing')
            # except (LabConfig.NoOptionError, LabConfig.NoSectionError):
                # integer_indexing = False
            # if integer_indexing:
                # df.set_index(['sequence_index', 'run number', 'run repeat'], inplace=True, drop=False)
            # else:
                # df.set_index([('sequence',) + padding,('run time',) + padding], inplace=True, drop=False)
                # df.index.names = ['sequence', 'run time']
        # except KeyError:
            # # Empty DataFrame or index column not found, so fall back to RangeIndex instead
            # pass
        # df.sort_index(inplace=True)
        return df
        
def globals_diff(run1, run2, group=None):
    return dict_diff(run1.get_globals(group), run2.get_globals(group))
 
class Run(object):
    def __init__(self,h5_path,no_write=False):
        self.no_write = no_write
        self._no_group = None
        self.h5_path = h5_path
        if not self.no_write:
            self._create_group_if_not_exists(h5_path, '/', 'results')
                     
        try:
            if not self.no_write:
                # The group were this run's results will be stored in the h5 file
                # will be the name of the python script which is instantiating
                # this Run object:
                frame = inspect.currentframe()
                __file__ = frame.f_back.f_globals['__file__']
                self.group = os.path.basename(__file__).split('.py')[0]
                self._create_group_if_not_exists(h5_path, 'results', self.group)
        except KeyError:
            # sys.stderr.write('Warning: to write results, call '
            # 'Run.set_group(groupname), specifying the name of the group '
            # 'you would like to save results to. This normally comes from '
            # 'the filename of your script, but since you\'re in interactive '
            # 'mode, there is no scipt name. Opening in read only mode for '
            # 'the moment.\n')
            
            # Backup the value of self.no_write for restoration once the group
            # is set
            self._no_group = (True, self.no_write)
            self.no_write = True
            
    def _create_group_if_not_exists(self, h5_path, location, groupname):
        """Creates a group in the HDF5 file at `location` if it does not exist.
        
        Only opens the h5 file in write mode if a group must be created.
        This ensures the last modified time of the file is only updated if
        the file is actually written to."""
        create_group = False
        with h5py.File(h5_path, 'r') as h5_file:
            if not groupname in h5_file[location]:
                create_group = True
        if create_group:
            with h5py.File(h5_path, 'r+') as h5_file:
                h5_file[location].create_group(groupname)

    def set_group(self, groupname):
        self.group = groupname
        self._create_group_if_not_exists(self.h5_path, '/', 'results')
        # restore no_write attribute now we have set the group
        if self._no_group is not None and self._no_group[0]:
            self.no_write = self._no_group[1]
            self._no_group = None

    def trace_names(self, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                try:
                    return list(h5_file['data']['traces'].keys())
                except KeyError:
                    return []
        else:
            try:
                return list(h5_file['data']['traces'].keys())
            except KeyError:
                return []
                
    def get_attrs(self, group, h5_file = None):
        """Returns all attributes of the specified group as a dictionary."""
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not group in h5_file:
                    raise Exception('The group \'%s\' does not exist'%group)
                return get_attributes(h5_file[group])
        else:
            if not group in h5_file:
                raise Exception('The group \'%s\' does not exist'%group)
            return get_attributes(h5_file[group])

    def get_trace(self,name, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not name in h5_file['data']['traces']:
                    raise Exception('The trace \'%s\' doesn not exist'%name)
                trace = h5_file['data']['traces'][name]
                return array(trace['t'],dtype=float),array(trace['values'],dtype=float)
        else:
            if not name in h5_file['data']['traces']:
                raise Exception('The trace \'%s\' doesn not exist'%name)
            trace = h5_file['data']['traces'][name]
            return array(trace['t'],dtype=float),array(trace['values'],dtype=float)            

    def get_time_array(self, device, name, frametype, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not device in h5_file['data']['time_arrays']:
                    raise Exception('No time arrays for device \'%s\''%device)
                if not name in h5_file['data']['time_arrays'][device]:
                    raise Exception('No time arrays named \'%s\''%name)
                if not frametype in h5_file['data']['time_arrays'][device][name]:
                    raise Exception('No time arrays with frametype \'%s\''%frametype)
                time_array = h5_file['data']['time_arrays'][device][name][frametype]
                return array(time_array,dtype=float)
        else:
            if not device in h5_file['data']['time_arrays']:
                raise Exception('No time arrays for device \'%s\''%device)
            if not name in h5_file['data']['time_arrays'][device]:
                raise Exception('No time arrays named \'%s\''%name)
            if not frametype in h5_file['data']['time_arrays'][device][name]:
                raise Exception('No time arrays with frametype \'%s\''%frametype)
            time_array = h5_file['data']['time_arrays'][device][name][frametype]
            return array(time_array,dtype=float)
            
    def get_time_array_RAW(self, device, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not device in h5_file['data']['time_arrays']:
                    raise Exception('No time arrays for device \'%s\''%device)
                time_array = h5_file['data']['time_arrays'][device]['arrival_times']
                sync_array = h5_file['data']['time_arrays'][device]['sync_times']
                return array(time_array,dtype=float),  array(sync_array,dtype=float)
        else:
            if not device in h5_file['data']['time_arrays']:
                raise Exception('No time arrays for device \'%s\''%device)
            time_array = h5_file['data']['time_arrays'][device]['arrival_times']
            sync_array = h5_file['data']['time_arrays'][device]['sync_times']
            return array(time_array,dtype=float),  array(sync_array,dtype=float)

    def get_result_array(self,group,name, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not group in h5_file['results']:
                    raise Exception('The result group \'%s\' doesn not exist'%group)
                if not name in h5_file['results'][group]:
                    raise Exception('The result array \'%s\' doesn not exist'%name)
                return array(h5_file['results'][group][name])
        else:
            if not group in h5_file['results']:
                raise Exception('The result group \'%s\' doesn not exist'%group)
            if not name in h5_file['results'][group]:
                raise Exception('The result array \'%s\' doesn not exist'%name)
            return array(h5_file['results'][group][name]) 
    
    def get_result(self, group, name, h5_file = None):
        """Return 'result' in 'results/group' that was saved by 
        the save_result() method."""
        
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                if not group in h5_file['results']:
                    raise Exception('The result group \'%s\' does not exist'%group)
                if not name in h5_file['results'][group].attrs.keys():
                    raise Exception('The result \'%s\' does not exist'%name)
                return get_attribute(h5_file['results'][group], name)
        else:
            if not group in h5_file['results']:
                raise Exception('The result group \'%s\' does not exist'%group)
            if not name in h5_file['results'][group].attrs.keys():
                raise Exception('The result \'%s\' does not exist'%name)
            return get_attribute(h5_file['results'][group], name)
        
    def get_results(self, group, *names, h5_file = None):
        """Iteratively call get_result(group,name) for each name provided.
        Returns a list of all results in same order as names provided."""
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                results = []
                for name in names:
                    results.append(self.get_result(group,name, h5_file = h5_file))
                return results
        else:
            results = []
            for name in names:
                results.append(self.get_result(group,name, h5_file = h5_file))
            return results
            
    def get_traces(self, *names, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                traces = []
                for name in names:
                    traces.extend(self.get_trace(name, h5_file = h5_file))
                return traces
        else:
            traces = []
            for name in names:
                traces.extend(self.get_trace(name, h5_file = h5_file))
            return traces
            
    def get_result_arrays(self, group, *names, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                results = []
                for name in names:
                    results.append(self.get_result_array(group, name, h5_file = h5_file))
                return results
        else:
            results = []
            for name in names:
                results.append(self.get_result_array(group, name, h5_file = h5_file))
            return results
        
    def get_time_arrays(self,device, name, *frametypes, h5_file = None):
        if h5_file == None:
            with h5py.File(self.h5_path, 'r') as h5_file:
                results = []
                for frametype in frametypes:
                    results.append(self.get_time_array(device, name, frametype, h5_file = h5_file))
                return results   
        else:
            results = []
            for frametype in frametypes:
                results.append(self.get_time_array(device, name, frametype, h5_file = h5_file))
            return results   
            
    def save_result(self, name, value, group=None, overwrite=True, h5_file = None):
        """Save a result to h5 file. Defaults are to save to the active group 
        in the 'results' group and overwrite an existing result.
        Note that the result is saved as an attribute of 'results/group' and
        overwriting attributes causes h5 file size bloat."""
        if self.no_write:
            raise Exception('This run is read-only. '
                            'You can\'t save results to runs through a '
                            'Sequence object. Per-run analysis should be done '
                            'in single-shot analysis routines, in which a '
                            'single Run object is used')
                            
        if h5_file == None:
            with h5py.File(self.h5_path,'a') as h5_file:
                if not group:
                    # Save to analysis results group by default
                    group = 'results/' + self.group
                elif not group in h5_file:
                    # Create the group if it doesn't exist
                    h5_file.create_group(group) 
                if name in h5_file[group].attrs and not overwrite:
                    raise Exception('Attribute %s exists in group %s. ' \
                                    'Use overwrite=True to overwrite.' % (name, group))                   
                set_attributes(h5_file[group], {name: value})
        else:
            if not group:
                # Save to analysis results group by default
                group = 'results/' + self.group
            elif not group in h5_file:
                # Create the group if it doesn't exist
                h5_file.create_group(group) 
            if name in h5_file[group].attrs and not overwrite:
                raise Exception('Attribute %s exists in group %s. ' \
                                'Use overwrite=True to overwrite.' % (name, group))                   
            set_attributes(h5_file[group], {name: value})
            
        if spinning_top:
            if self.h5_path not in _updated_data:
                _updated_data[self.h5_path] = {}
            if group.startswith('results'):
                toplevel = group.replace('results/', '', 1)
                _updated_data[self.h5_path][toplevel, name] = value

    def save_result_array(self, name, data, group=None, 
                          overwrite=True, keep_attrs=False,h5_file = None, **kwargs):
        """Save data array to h5 file. Defaults are to save to the active 
        group in the 'results' group and overwrite existing data.
        Additional keyword arguments are passed directly to h5py.create_dataset()."""
        if self.no_write:
            raise Exception('This run is read-only. '
                            'You can\'t save results to runs through a '
                            'Sequence object. Per-run analysis should be done '
                            'in single-shot analysis routines, in which a '
                            'single Run object is used')
        if h5_file == None:
            with h5py.File(self.h5_path, 'a') as h5_file:
                attrs = {}
                if not group:
                    # Save dataset to results group by default
                    group = 'results/' + self.group
                elif not group in h5_file:
                    # Create the group if it doesn't exist
                    h5_file.create_group(group) 
                if name in h5_file[group]:
                    if overwrite:
                        # Overwrite if dataset already exists
                        if keep_attrs:
                            attrs = dict(h5_file[group][name].attrs)
                        del h5_file[group][name]
                    else:
                        raise Exception('Dataset %s exists. Use overwrite=True to overwrite.' % 
                                         group + '/' + name)
                h5_file[group].create_dataset(name, data=data, **kwargs)
                for key, val in attrs.items():
                    h5_file[group][name].attrs[key] = val
        else:
            attrs = {}
            if not group:
                # Save dataset to results group by default
                group = 'results/' + self.group
            elif not group in h5_file:
                # Create the group if it doesn't exist
                h5_file.create_group(group) 
            if name in h5_file[group]:
                if overwrite:
                    # Overwrite if dataset already exists
                    if keep_attrs:
                        attrs = dict(h5_file[group][name].attrs)
                    del h5_file[group][name]
                else:
                    raise Exception('Dataset %s exists. Use overwrite=True to overwrite.' % 
                                     group + '/' + name)
            h5_file[group].create_dataset(name, data=data, **kwargs)
            for key, val in attrs.items():
                h5_file[group][name].attrs[key] = val
        
    def save_results(self, *args, print_out=True,**kwargs):
        """Iteratively call save_result() on multiple results.
        Assumes arguments are ordered such that each result to be saved is
        preceeded by the name of the attribute to save it under.
        Keywords arguments are passed to each call of save_result()."""
        names = args[::2]
        values = args[1::2]
        with h5py.File(self.h5_path, 'a') as h5_file:
            for name, value in zip(names, values):
                if print_out :
                    print('saving %s =' % name, value)
                self.save_result(name, value, h5_file = h5_file, **kwargs)
            
    def save_results_dict(self, results_dict, uncertainties=False, h5_file = None, **kwargs):
        if h5_file == None:
            with h5py.File(self.h5_path, 'a') as h5_file:
                for name, value in results_dict.items():
                    if not uncertainties:
                        self.save_result(name, value,h5_file = h5_file, **kwargs)
                    else:
                        self.save_result(name, value[0], h5_file = h5_file, **kwargs)
                        self.save_result('u_' + name, value[1], h5_file = h5_file, **kwargs)
        else:
            for name, value in results_dict.items():
                if not uncertainties:
                    self.save_result(name, value,h5_file = h5_file, **kwargs)
                else:
                    self.save_result(name, value[0], h5_file = h5_file, **kwargs)
                    self.save_result('u_' + name, value[1], h5_file = h5_file, **kwargs)  
                    
    def save_result_arrays_dict(self, result_arrays_dict,h5_file = None, **kwargs):
        if h5_file == None:
            with h5py.File(self.h5_path, 'a') as h5_file:
                for name, value in result_arrays_dict.items():
                    self.save_result_array(name, value,h5_file = h5_file, **kwargs)
        else:
            for name, value in result_arrays_dict.items():
                self.save_result_array(name, value,h5_file = h5_file, **kwargs)
                
    def save_total_results_dict(self, total_results_dict):
        with h5py.File(self.h5_path, 'a') as h5_file:
            self.save_result_arrays_dict(total_results_dict['array'], h5_file = h5_file)
            self.save_results_dict(total_results_dict['single'], h5_file = h5_file)
    
    def save_result_arrays(self, *args, **kwargs):
        """Iteratively call save_result_array() on multiple data sets. 
        Assumes arguments are ordered such that each dataset to be saved is 
        preceeded by the name to save it as. 
        All keyword arguments are passed to each call of save_result_array()."""
        names = args[::2]
        values = args[1::2]
        with h5py.File(self.h5_path, 'a') as h5_file:
            for name, value in zip(names, values):
                self.save_result_array(name, value, h5_file = h5_file, **kwargs)
    
    def get_image(self,orientation,label,image):
        with h5py.File(self.h5_path, 'r') as h5_file:
            if not 'images' in h5_file:
                raise Exception('File does not contain any images')
            if not orientation in h5_file['images']:
                raise Exception('File does not contain any images with orientation \'%s\''%orientation)
            if not label in h5_file['images'][orientation]:
                raise Exception('File does not contain any images with label \'%s\''%label)
            if not image in h5_file['images'][orientation][label]:
                raise Exception('Image \'%s\' not found in file'%image)
            return array(h5_file['images'][orientation][label][image])
    
    def get_images(self,orientation,label, *images):
        results = []
        for image in images:
            results.append(self.get_image(orientation,label,image))
        return results
        
    def get_all_image_labels(self):
        images_list = {}
        with h5py.File(self.h5_path, 'r') as h5_file:
            for orientation in h5_file['/images'].keys():
                images_list[orientation] = list(h5_file['/images'][orientation].keys())               
        return images_list                
    
    def get_image_attributes(self, orientation):
        with h5py.File(self.h5_path, 'r') as h5_file:
            if not 'images' in h5_file:
                raise Exception('File does not contain any images')
            if not orientation in h5_file['images']:
                raise Exception('File does not contain any images with orientation \'%s\''%orientation)
            return get_attributes(h5_file['images'][orientation])

    def get_globals(self,group=None):
        if not group:
            with h5py.File(self.h5_path, 'r') as h5_file:
                return dict(h5_file['globals'].attrs)
        else:
            try:
                with h5py.File(self.h5_path, 'r') as h5_file:
                    return dict(h5_file['globals'][group].attrs)
            except KeyError:
                return {}

    def get_globals_raw(self, group=None):
        globals_dict = {}
        with h5py.File(self.h5_path, 'r') as h5_file:
            if group == None:
                for obj in h5_file['globals'].values():
                    temp_dict = dict(obj.attrs)
                    for key, val in temp_dict.items():
                        globals_dict[key] = val
            else:
                globals_dict = dict(h5_file['globals'][group].attrs)
        return globals_dict
        
    # def iterable_globals(self, group=None):
        # raw_globals = self.get_globals_raw(group)
        # print raw_globals.items()
        # iterable_globals = {}
        # for global_name, expression in raw_globals.items():
            # print expression
            # # try:
                # # sandbox = {}
                # # exec('from pylab import *',sandbox,sandbox)
                # # exec('from runmanager.functions import *',sandbox,sandbox)
                # # value = eval(expression,sandbox)
            # # except Exception as e:
                # # raise Exception('Error parsing global \'%s\': '%global_name + str(e))
            # # if isinstance(value,types.GeneratorType):
               # # print global_name + ' is iterable.'
               # # iterable_globals[global_name] = [tuple(value)]
            # # elif isinstance(value, ndarray) or  isinstance(value, list):
               # # print global_name + ' is iterable.'            
               # # iterable_globals[global_name] = value
            # # else:
                # # print global_name + ' is not iterable.'
            # return raw_globals
            
    def get_globals_expansion(self):
        expansion_dict = {}
        def append_expansion(name, obj):
            if 'expansion' in name:
                temp_dict = dict(obj.attrs)
                for key, val in temp_dict.items():
                    if val:
                        expansion_dict[key] = val
        with h5py.File(self.h5_path, 'r') as h5_file:
            h5_file['globals'].visititems(append_expansion)
        return expansion_dict
                   
    def get_units(self, group=None):
        units_dict = {}
        def append_units(name, obj):
            if 'units' in name:
                temp_dict = dict(obj.attrs)
                for key, val in temp_dict.items():
                    units_dict[key] = val
        with h5py.File(self.h5_path, 'r') as h5_file:
            h5_file['globals'].visititems(append_units)
        return units_dict

    def globals_groups(self):
        with h5py.File(self.h5_path, 'r') as h5_file:
            try:
                return list(h5_file['globals'].keys())
            except KeyError:
                return []   
                
    def globals_diff(self, other_run, group=None):
        return globals_diff(self, other_run, group)            
    
        
class Sequence(Run):
    def __init__(self, h5_path, run_paths, no_write=False):
        if isinstance(run_paths, pandas.DataFrame):
            run_paths = run_paths['filepath']
        self.h5_path = h5_path
        self.no_write = no_write
        if not self.no_write:
            self._create_group_if_not_exists(h5_path, '/', 'results')
                 
        self.runs = {path: Run(path,no_write=True) for path in run_paths}
        
        # The group were the results will be stored in the h5 file will
        # be the name of the python script which is instantiating this
        # Sequence object:
        frame = inspect.currentframe()
        try:
            __file__ = frame.f_back.f_locals['__file__']
            self.group = os.path.basename(__file__).split('.py')[0]
            if not self.no_write:
                self._create_group_if_not_exists(h5_path, 'results', self.group)
        except KeyError:
            sys.stderr.write('Warning: to write results, call '
            'Sequence.set_group(groupname), specifying the name of the group '
            'you would like to save results to. This normally comes from '
            'the filename of your script, but since you\'re in interactive '
            'mode, there is no scipt name. Opening in read only mode for '
            'the moment.\n')
            self.no_write = True
        
    def get_trace(self,*args):
        return {path:run.get_trace(*args) for path,run in self.runs.items()}
        
    def get_result_array(self,*args):
        return {path:run.get_result_array(*args) for path,run in self.runs.items()}
         
    def get_traces(self,*args):
        raise NotImplementedError('If you want to use this feature please ask me to implement it! -Chris')
             
    def get_result_arrays(self,*args):
        raise NotImplementedError('If you want to use this feature please ask me to implement it! -Chris')
     
    def get_image(self,*args):
        raise NotImplementedError('If you want to use this feature please ask me to implement it! -Chris')     


def figure_to_clipboard(figure=None, **kwargs):
    """Copy a matplotlib figure to the clipboard as a png. If figure is None,
    the current figure will be copied. Copying the figure is implemented by
    calling figure.savefig() and then copying the image data from the
    resulting file. Any keyword arguments will be passed to the call to
    savefig(). If bbox_inches keyword arg is not provided,
    bbox_inches='tight' will be used"""
    
    import matplotlib.pyplot as plt
    from zprocess import start_daemon
    import tempfile

    if not 'bbox_inches' in kwargs:
        kwargs['bbox_inches'] = 'tight'
               
    if figure is None:
        figure = plt.gcf()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tempfile_name = f.name

    figure.savefig(tempfile_name, **kwargs)

    tempfile2clipboard = os.path.join(LYSE_DIR, 'tempfile2clipboard.py')
    start_daemon([sys.executable, tempfile2clipboard, '--delete', tempfile_name])


def register_plot_class(identifier, cls):
    if not spinning_top:
        msg = """Warning: lyse.register_plot_class has no effect on scripts not run with
            the lyse GUI.
            """
        sys.stderr.write(dedent(msg))
    _plot_classes[identifier] = cls

def get_plot_class(identifier):
    return _plot_classes.get(identifier, None)

def delay_results_return():
    global _delay_flag
    if not spinning_top:
        msg = """Warning: lyse.delay_results_return has no effect on scripts not run 
            with the lyse GUI.
            """
        sys.stderr.write(dedent(msg))
    _delay_flag = True
