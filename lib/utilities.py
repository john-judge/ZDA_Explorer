#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os, glob
import struct
import numpy as np
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from scipy.ndimage import gaussian_filter

import scipy.cluster.hierarchy as shc
import itertools
#from scipy.interpolate import griddata

from pprint import pprint

# for SNR
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# ZDA Explorer modules
from lib.trace import Tracer
from lib.image import MovieMaker, SignalProcessor
from lib.freq import FreqAnalyzer
from lib.snr import AnalyzerSNR


############################# Data load functions ##########################
############################################################################

class Dataset:
    
    def __init__(self, filename, x_range=[0,-1], y_range=[0,-1], t_range=[0,-1]):
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.data, metadata, self.rli = self.read_zda_to_df(filename)
        self.filename = filename
        self.meta = metadata
        if self.meta is not None:
            self.version = metadata['version']
            self.slice_number = metadata['slice_number']
            self.location_number = metadata['location_number']
            self.record_number = metadata['record_number'] 
            self.camera_program = metadata['camera_program'] 
            self.number_of_trials = metadata['number_of_trials']
            self.interval_between_trials = metadata['interval_between_trials']
            self.acquisition_gain = metadata['acquisition_gain']
            self.points_per_trace = metadata['points_per_trace']
            self.time_RecControl = metadata['time_RecControl']
            self.reset_onset = metadata['reset_onset']
            self.reset_duration =  metadata['reset_duration']
            self.shutter_onset = metadata['shutter_onset'] 
            self.shutter_duration = metadata['shutter_duration']
            self.stimulation1_onset = metadata['stimulation1_onset']
            self.stimulation1_duration = metadata['stimulation1_duration']
            self.stimulation2_onset = metadata['stimulation2_onset'] 
            self.stimulation2_duration = metadata['stimulation2_duration']
            self.acquisition_onset = metadata['acquisition_onset'] 
            self.interval_between_samples = metadata['interval_between_samples']
            self.raw_width = metadata['raw_width']
            self.raw_height = metadata['raw_height']
            
            # original dimensions
            self.original = {'raw_width': self.raw_width,
                             'raw_height': self.raw_height,
                             'points_per_trace': self.points_per_trace }
           
    def clip_data(self, x_range=[0,-1], y_range=[0,-1], t_range=[0,-1]):
        """ Imposes a range restriction on frames and/or traces """
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        
    def get_unclipped_data(self, trial=None):
        """ Returns unclipped data """
        
        # Reset to unclipped values
        self.meta['points_per_trace'] = self.original['points_per_trace']
        self.meta['raw_width'] = self.original['raw_width']
        self.meta['raw_height'] = self.original['raw_height']
        
        self.raw_width = self.meta['raw_width']
        self.raw_height = self.meta['raw_height']
        self.points_per_trace = self.meta['points_per_trace']

        if trial is not None:
            return self.data[trial,:,:,:]
        return self.data
    
    def get_data(self, trial=None):
        """ Returns clipped data """
        
        # Set to clipped values
        self.meta['points_per_trace'] = self.t_range[1] - self.t_range[0]
        self.meta['raw_width'] = self.x_range[1] - self.x_range[0]
        self.meta['raw_height'] = self.y_range[1] - self.y_range[0]
        
        self.raw_width = self.meta['raw_width']
        self.raw_height = self.meta['raw_height']
        self.points_per_trace = self.meta['points_per_trace']
        
        
        if trial is not None:
            return self.data[trial,
                    self.x_range[0]:self.x_range[1],
                    self.y_range[0]:self.y_range[1],
                    self.t_range[0]:self.t_range[1]]
        else:
            return self.data[:,
                    self.x_range[0]:self.x_range[1],
                    self.y_range[0]:self.y_range[1],
                    self.t_range[0]:self.t_range[1]]
        
    def get_meta(self):
        """ Returns metadata dictionary. Mostly for legacy behavior. """
        return self.meta
    
    def get_rli(self):
        """ Returns RLI data """
        return self.rli
    
    def get_trial_data(self, trial_no):
        """ Returns array slice for trial number """
        return 
    

    def read_zda_to_df(self, zda_file):
        ''' Reads ZDA file to dataframe, and returns
        metadata as a dict. 
        ZDA files are a custom PhotoZ binary format that must be interpreted byte-
        by-byte'''
        file = open(zda_file, 'rb')
        print(zda_file)
        # data type sizes in bytes
        chSize = 1
        shSize = 2
        nSize = 4
        tSize = 8
        fSize = 4

        metadata = {}
        metadata['version'] = (file.read(chSize))
        metadata['slice_number'] = (file.read(shSize))
        metadata['location_number'] = (file.read(shSize))
        metadata['record_number'] = (file.read(shSize))
        metadata['camera_program'] = (file.read(nSize))

        metadata['number_of_trials'] = (file.read(chSize))
        metadata['interval_between_trials'] = (file.read(chSize))
        metadata['acquisition_gain'] = (file.read(shSize))
        metadata['points_per_trace'] = (file.read(nSize))
        metadata['time_RecControl'] = (file.read(tSize))

        metadata['reset_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['reset_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['stimulation1_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation1_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['acquisition_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['interval_between_samples'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['raw_width'] = (file.read(nSize))
        metadata['raw_height'] = (file.read(nSize))

        # Bytes to Python data type
        for k in metadata:
            if k in ['interval_between_samples',] or 'onset' in k or 'duration' in k:
                pass # any additional float processing can go here
            elif k == 'time_RecControl':
                pass # TO DO: timestamp processing
            else:
                metadata[k] = int.from_bytes(metadata[k], "little") # endianness

        num_diodes = metadata['raw_width'] * metadata['raw_height']

        file.seek(1024, 0)
        # RLI 
        rli = {}
        rli['rli_low'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_high'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_max'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]

        raw_data = np.zeros((metadata['number_of_trials'],
                             metadata['raw_width'],
                             metadata['raw_height'],
                             metadata['points_per_trace'])).astype(int)

        for i in range(metadata['number_of_trials']):
            for jw in range(metadata['raw_width']):
                for jh in range(metadata['raw_height']):
                    for k in range(metadata['points_per_trace']):
                        pt = file.read(shSize)
                        if not pt:
                            print("Ran out of points.",len(raw_data))
                            file.close()
                            return metadata
                        raw_data[i,jw,jh,k] = int.from_bytes(pt, "little")

        file.close()
        return raw_data, metadata, rli


class DataLoader:
    
    def __init__(self):
        self.all_data = {} # maps file names to Dataset objects
        
    def select_data_by_keyword(self, keyword):
        """ Returns the data for the first file matching the keyword """
        for file in self.all_data:
            if keyword in file:
                return self.all_data[file]

    def load_all_zda(self, data_dir='.'):
        ''' Loads all ZDA data in data_dir into a dictionary of dataframes and metadata '''
        n_files_loaded = 0
        for dirName, subdirList, fileList in os.walk(data_dir,topdown=True):
            for file in fileList:
                file = str(dirName + "/" + file)
                if '.zda' == file[-4:]:
                    self.all_data[file] = Dataset(file)
                    n_files_loaded += 1

        print('Number of files loaded:', n_files_loaded)
        return self.all_data
    
    def get_dataset(self, filename):
        return self.all_data[filename]
    