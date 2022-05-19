import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.polynomial import polynomial

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

class Tracer:
    
    def plot_trace(self, raw_data, x, y, interval, trial=None, reg=None):
        ''' View a single trace '''
        time = [interval * i for i in range(raw_data.shape[1]) ]

        fig, ax = plt.subplots()
        if trial is not None:
            ax.plot(time,
                    raw_data[trial,:,x,y],
                    color='red')
        else:
            ax.plot(time,
                    raw_data[:,x,y],
                    color='red')
        if reg is not None:
            ax.plot(time, reg, color='blue', linewidth=3)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.grid(True)
        plt.show()

    def polynomial_subtract_noise(self, raw_data, trial, jw, jh, interval, linear=True, plot=False, skip_window=None):

        t = np.array([interval * i for i in range(raw_data.shape[1])])
        t = t.reshape(-1)
        y = None
        if trial is None:
            y = raw_data[:, jw, jh]
        else:
            y = raw_data[trial, :, jw, jh]
        y = y.reshape(-1)

        power = 8
        coeffs, stats = polynomial.polyfit(t, y, power, full=True)
        reg = np.array(polynomial.polyval(t, coeffs))

        if plot:
            self.plot_trace(raw_data, jw, jh, interval, trial=trial, reg=reg)

        if trial is not None:
            raw_data[trial, :, jw, jh] = (raw_data[trial, :, jw, jh].reshape(-1, 1)
                                          - reg.reshape(-1,1)).reshape(-1)
        else:
            raw_data[:, jw, jh] = (raw_data[:, jw, jh].reshape(-1, 1)
                                   - reg.reshape(-1,1)).reshape(-1)

    def subtract_noise(self, raw_data, trial, jw, jh, interval, reg_type=1, plot=False):
        """ subtract background drift off of single trace
        Reg type: 1 = linear, 2 = exp, 3 = poly-8 """
        if reg_type == 3:
            return self.polynomial_subtract_noise(raw_data, trial, jw, jh, interval, plot=plot)
        linear = (reg_type == 1)
        X = np.array([interval * i for i in range(raw_data.shape[1]) ])
        X = X.reshape(-1,1)
        y = None
        if trial is None:
            y = raw_data[:, jw, jh]
        else:
            y = raw_data[trial, :, jw, jh]
        if not linear: # then Exponential Regression
            y[y <= 0] = 1
            y = np.log(y)
        y = y.reshape(-1,1)
        reg = LinearRegression().fit(X, y).predict(X)

        if not linear: # then Exponential Regression
            reg = np.exp(reg)

        if plot:
            self.plot_trace(raw_data, jw, jh, interval, trial=trial, reg=reg)
            
        if trial is not None:
            raw_data[trial, :, jw, jh] = (raw_data[trial, :, jw, jh].reshape(-1, 1)
                                      - reg).reshape(-1)
        else:
            raw_data[:, jw, jh] = (raw_data[:, jw, jh].reshape(-1, 1)
                                      - reg).reshape(-1)

    def correct_background(self, meta, raw_data, trial_dim=True):
        """ subtract background drift off of all traces """
        if trial_dim:
            for i in range(raw_data.shape[0]):
                for jw in range(raw_data.shape[2]):
                    for jh in range(raw_data.shape[3]):
                        self.subtract_noise(raw_data, i, jw, jh,
                                              meta['interval_between_samples'],
                                              plot=(i==0 and jw==0 and jh==0),
                                              reg_type=3)
        else:
            for jw in range(raw_data.shape[1]):
                for jh in range(raw_data.shape[2]):
                    self.subtract_noise(raw_data, None, jw, jh,
                                          meta['interval_between_samples'],
                                          plot=False,
                                          reg_type=3)


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
