import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from skimage.measure import block_reduce
import cv2 as cv



class MovieMaker:

    def imageFolder2mpeg(self, input_path, output_path='./output_video.mpeg', fps=30.0):
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

    def create_window_snr_movie(self, filtered_data, len_window=8, x_range=[0,-1], y_range=[0,-1], data_dir=''):
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

class SignalProcessor:

    def create_binned_data(self, data, binning_factor=2):
        ''' binning utility function '''
        if binning_factor < 2:
            return data
        return block_reduce(data, 
                            (1, binning_factor, binning_factor, 1),
                            np.average)

    def filter_temporal(self, meta, raw_data, sigma_t = 1.0):
        ''' Temporal filtering: 1-d binomial 8 filter (approx. Gaussian) '''
        filtered_data = np.zeros(raw_data.shape)
        for i in range(meta['number_of_trials']):
            for jw in range(meta['raw_width']):
                for jh in range(meta['raw_height']):
                    filtered_data[i, jw, jh, :] = gaussian_filter(raw_data[i, jw, jh, :], 
                                                    sigma=sigma_t)
        return filtered_data

    def filter_spatial(self, meta, raw_data, sigma_s = 1.0):
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
