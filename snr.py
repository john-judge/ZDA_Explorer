import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

class AnalyzerSNR:
    
    def __init__(self, data):
        self.data = data
        self.snr = None
        
    
    def get_snr(self, plot=False):
        """ Given a single trial, compute the SNR image for this trial """
        self.snr = np.mean(self.data, axis=2) / np.std(self.data, axis=2)
        
        if plot:
            plt.imshow(self.snr, cmap='jet', interpolation='nearest')
            plt.show()
            
        return self.snr
    
    def cluster_on_snr(self, k_clusters=3, snr_cutoff=0.7, plot=False):
        """ Perform 1-D clustering on SNR after masking out the pixels
        whose snr is below snr_cutoff (a percentile in range [0,1]) """
        
        if self.snr is None:
            raise ValueError("No SNR data found.")
            
        snr_cutoff = np.percentile(self.snr, snr_cutoff * 100)
        mask = (self.snr >= snr_cutoff).astype(np.float)
        if plot:
            # masked image: reasonability check
            plt.imshow(snr * mask, cmap='jet', interpolation='nearest')
            plt.show()

        km = KMeans(n_clusters=k_clusters+1).fit(snr.reshape(-1,1)) # +1 for the masked 0's
        
        clustered = np.array(km.labels_).reshape(self.snr.shape) + 1
        clustered = clustered.astype(np.float)
        
        if plot:
            plt.imshow(clustered * mask, cmap='viridis', interpolation='nearest')
            plt.show()


    
    