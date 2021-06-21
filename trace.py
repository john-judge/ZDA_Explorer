import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

class Tracer:
    
    def plot_trace(self, raw_data, x, y, interval, trial=None, reg=None):
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

    def subtract_noise(self, raw_data, trial, jw, jh, interval, linear=True, plot=False):
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
            self.plot_trace(raw_data, jw, jh, interval, trial=trial, reg=reg)

        raw_data[trial, jw, jh, :] = (raw_data[trial, jw, jh, :].reshape(-1,1) 
                                      - reg).reshape(-1)

    def correct_background(self, meta, raw_data):
        """ subtract background drift off of all traces """
        for i in range(meta['number_of_trials']):
            for jw in range(meta['raw_width']):
                for jh in range(meta['raw_height']):
                    self.subtract_noise(raw_data, i, jw, jh, 
                                          meta['interval_between_samples'],
                                          plot=(i==0 and jw==40 and jh == 40),
                                          linear=False )

    def get_half_width(self, location, trace):
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
