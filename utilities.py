#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os, glob
import struct
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_samples, silhouette_score
from skimage.measure import block_reduce

import scipy.cluster.hierarchy as shc
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq, fftshift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import itertools
from scipy.interpolate import griddata

from yellowbrick.cluster import KElbowVisualizer
from pprint import pprint
import cv2 as cv


############################# Utility functions ############################
############################################################################

def load_all_zda(data_dir='.'):
    ''' Loads all ZDA data in data_dir into a dictionary of dataframes and metadata '''
    all_data = {}
    n_files_loaded = 0
    for dirName, subdirList, fileList in os.walk(data_dir,topdown=True):
        for file in fileList:
            file = str(dirName + "/" + file)
            if '.zda' == file[-4:]:
                raw_data, meta, rli = read_zda_to_df(file)
                all_data[file, 'data'] = raw_data
                all_data[file, 'meta'] = meta
                all_data[file, 'rli'] = rli
                n_files_loaded += 1

    print('Number of files loaded:', n_files_loaded)
    return all_data

def read_zda_to_df(zda_file):
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
    
    #count = 0
    #while pt:
    #    pt = file.read(shSize)
    #    count += 1
    #print('left over:', count)
    
    file.close()
    return raw_data, metadata, rli
    
def plot_trace(raw_data, x, y, interval, trial=None, reg=None):
    ''' View a single trace '''
    time = [interval * i for i in range(raw_data.shape[-1]) ]
    
    if trial is not None:
        plt.plot(time, 
                raw_data[trial,x,y,:], 
                color='red')
    else:
        plt.plot(time, 
                raw_data[x,y,:], 
                color='red')
    if reg is not None:
        plt.plot(time, reg, color='blue', linewidth=3)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage")
    plt.grid(True)
    plt.show()

def subtract_noise(raw_data, trial, jw, jh, interval, linear=True, plot=False):
    """ subtract background drift off of single trace """
    X = np.array([interval * i for i in range(raw_data.shape[3]) ]).reshape(-1,1)
    y = raw_data[trial, jw, jh, :]
    if not linear: # then Exponential Regression
        y[y <= 0] = 1
        y = np.log(y)
    y = y.reshape(-1,1)
    reg = LinearRegression().fit(X, y).predict(X)

    if not linear: # then Exponential Regression
        reg = np.exp(reg)

    if plot:
        plot_trace(raw_data, jw, jh, interval, trial=trial, reg=reg)
    
    raw_data[trial, jw, jh, :] = (raw_data[trial, jw, jh, :].reshape(-1,1) 
                                  - reg).reshape(-1)

def correct_background(meta, raw_data):
    """ subtract background drift off of all traces """
    for i in range(meta['number_of_trials']):
        for jw in range(meta['raw_width']):
            for jh in range(meta['raw_height']):
                subtract_noise(raw_data, i, jw, jh, 
                                      meta['interval_between_samples'],
                                      plot=(i==0 and jw==40 and jh == 40),
                                      linear=False )

def get_half_width(location, trace):
    """ Return TRACE's zeros on either side, if any, of location 
        Includes linear interpolation """
    hw = trace[location] / 2
    if len(trace.shape) < 1 or trace.shape[0] < 2:
        return 0

    i_left = location
    while trace[i_left] > hw :
        i_left -= 1
        if i_left < 0:
            return None
    # linearly interpolate
    i_left -= trace[i_left+1] / (trace[i_left+1] + trace[i_left])

    i_right = location
    while trace[i_right] > hw:
        i_right += 1
        if i_right >= trace.shape[0]:
            return None
    # linearly interpolate
    i_right += trace[i_right-1] / (trace[i_right-1] + trace[i_right])

    if i_right - i_left <= 0:
        return None
    return i_right - i_left

def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''
    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'MPG1')
    video_writer = cv.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:

        frame_i = cv.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()

def create_window_snr_movie(filtered_data, len_window=8, x_range=[0,-1], y_range=[0,-1], data_dir=''):
    ''' Windowed-SNR movies: Pool SNR over a sliding window '''
    trial = filtered_data[0, x_range[0]:x_range[1], 
                        y_range[0]:y_range[1], :] 

    x,y,t = trial.shape
    windowed_snr_data = np.zeros((x, y, t - len_window))

    vrange = [np.min(trial) + 0.0000001, np.max(trial)]

    # clean old JPGs in target dir
    fileList = glob.glob(data_dir + "*.jpg")    
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    for start_frame in range(t - len_window):
        window = trial[:,:,start_frame:start_frame+len_window]
        windowed_snr_data[:,:,start_frame] = np.mean(window, axis=2) / np.std(window, axis=2)

        # min-max normalization over entire video, not just frame
        window = (window - vrange[0]) / (vrange[1] - vrange[0])

        plt.imshow(windowed_snr_data[:,:,start_frame], 
                  cmap='jet', 
                  interpolation='nearest')
        fname = str(start_frame)
        while len(fname) < 5:
            fname = "0" + fname
        plt.savefig(data_dir + fname + ".jpg")

def create_binned_data(data, binning_factor=2):
    ''' binning utility function '''
    if binning_factor < 2:
        return data
    return block_reduce(data, 
                        (1, binning_factor, binning_factor, 1),
                        np.average)

def filter_temporal(meta, raw_data, sigma_t = 1.0):
    ''' Temporal filtering: 1-d binomial 8 filter (approx. Gaussian) '''
    filtered_data = np.zeros(raw_data.shape)
    for i in range(meta['number_of_trials']):
        for jw in range(meta['raw_width']):
            for jh in range(meta['raw_height']):
                filtered_data[i, jw, jh, :] = gaussian_filter(raw_data[i, jw, jh, :], 
                                                sigma=sigma_t)
    return filtered_data

def filter_spatial(meta, raw_data, sigma_s = 1.0):
    ''' Spatial filtering: Gaussian '''
    filtered_data = np.zeros(raw_data.shape)
    raw_data = raw_data
    filter_size = int(sigma_s * 3.5)
    for i in range(meta['number_of_trials']):
        for t in range(raw_data.shape[3]):
            filtered_data[i, :, :, t] = cv.GaussianBlur(raw_data[i, :, :, t].astype(np.float32),
                                                        (filter_size,filter_size),
                                                        sigma_s)
    return filtered_data

def compute_fft_binning(meta):
    ''' Compute a list of frequencies to use as freq domain for FFT '''
    sampling_rate = 1000 / meta['interval_between_samples'] # Hz
    x_fft = fftfreq(meta['points_per_trace'], 1 / sampling_rate) 
    x_fft = fftshift(x_fft)
    return x_fft

def decompose_trace_frequencies(meta, trace, x_fft=None, lower_freq=0, upper_freq=300, y_max=2000):
    ''' Plots the frequencies and returns the FFT of the trace 
        https://realpython.com/python-scipy-fft/#the-scipyfft-module
    '''
    if x_fft is None:
        x_fft = compute_fft_binning(meta)
        
    y_freq_transform = np.abs(fft(trace))
    
    # get rid of line joining last and first points
    y_freq_transform = fftshift(y_freq_transform)
    
    plt.plot(x_fft, 
             y_freq_transform)
    plt.xlim([lower_freq, upper_freq])
    plt.ylim([0,y_max])
    plt.show()
    
    return y_freq_transform
