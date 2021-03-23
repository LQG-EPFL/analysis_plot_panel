# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:08:46 2021

@author: Nick Sauerwein
"""

import lyse
import numpy as np

import os.path
import time

import h5py

def get_mtime(filename):
    return time.ctime(os.path.getmtime(filename))

class DataExtractorManager:
    
    def __init__(self):
        
        self.data_extractors = {}
        
    def update_local_data(self, h5_path):
        with h5py.File(h5_path, 'r') as h5_file:
            for key in self.data_extractors:
                self.data_extractors[key].update_local_data(h5_path, h5_file = h5_file)
            
    def clean_memory(self, h5_paths):
        for key in self.data_extractors:
            self.data_extractors[key].clean_memory(h5_paths)
            
    def __getitem__(self, key):
        return self.data_extractors[key]
    
    def __setitem__(self, key, de):
        self.data_extractors[key] = de
        

class DataExtractor:
    def __init__(self, load_to_ram = True):
        
        self.load_to_ram = load_to_ram
        
        self.local_datas = {}
        self.local_mtimes = {}
        
        if self.load_to_ram:
            self.local_data_changed = True          
        
    def update_local_data(self, h5_path, h5_file = None):
        if h5_path in self.local_datas and self.local_mtimes[h5_path] == get_mtime(h5_path):
            self.local_data_changed = False
        elif self.load_to_ram:
            self.local_datas[h5_path] = self.extract_data(h5_path, h5_file = h5_file)
            self.local_mtimes[h5_path] = get_mtime(h5_path)
            self.local_data_changed = True
            
    def update_local_datas(self):
        for key in self.local_datas:
            self.update_local_data(key)
    
    def get_data(self, h5_path, h5_file = None):
        
        if self.load_to_ram:
            
            self.update_local_data(h5_path)
            
            return self.local_datas[h5_path]        
        else:
            return self.extract_data(h5_path, h5_file = h5_file)
        
    def clean_memory(self, h5_paths):
        for key in list(self.local_datas):
            if key not in h5_paths.to_list():
                del self.local_datas[key]
                del self.local_mtimes[key]
                
                self.local_data_changed = True        
        self.update_local_datas()
            
class MultiDataExtractor(DataExtractor):
    def __init__(self,**kwargs):
        
        super().__init__(load_to_ram=False, **kwargs)
        
        self.data_extractors = {}
        self.children_changed = False
        
    def extract_data(self, h5_path, h5_file = None):
        
        data = {}
        
        for key in self.data_extractors:
            data[key] = self.data_extractors[key].get_data(h5_path, h5_file = h5_file)
        
        self.children_changed = False
        
        return [data]
    
    def clean_children(self, keys):
        self.children_changed = False
        for key in list(self.data_extractors):
            if key not in keys:
                del self.data_extractors[key]
                self.children_changed = True
                
    def clean_memory(self, h5_paths):
       for key in self.data_extractors:
            self.data_extractors[key].clean_memory(h5_paths)
            
    def __getitem__(self, key):
        return self.data_extractors[key]
    
    def __setitem__(self, key, de):
        self.children_changed = True
        self.data_extractors[key] = de
        
    @property
    def local_data_changed(self):
        return any([self.data_extractors[key].local_data_changed for key in self.data_extractors]) or self.children_changed
        
class EmptyDataExtractor(DataExtractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
    def extract_data(self, h5_path, h5_file = None):
        return []     
    
class ArrayDataExtractor(DataExtractor):

    def __init__(self,idx,  **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        
    def extract_data(self, h5_path, h5_file = None):
        run = lyse.Run(h5_path)
        try:
            res = run.get_result_array(self.idx[0],self.idx[1], h5_file = h5_file)
        except:
            res = None
        
        return res

class SingleDataExtractor(DataExtractor):

    def __init__(self,idx,  **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        
    def extract_data(self, h5_path, h5_file = None):
        run = lyse.Run(h5_path)
        try:
            res = run.get_result(self.idx[0],self.idx[1], h5_file = h5_file)
        except:
            res = None
        
        return res 

