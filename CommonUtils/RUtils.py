# Some common tools to go from
# ROOT -> array-type data 

# http://scikit-hep.org/root_numpy/
# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.hist2array.html

from ROOT import TH1, TH2, TFile
import numpy as np
from root_numpy import hist2array

def root2np(file_path="data.root", hist_path="Tracks/pvalue", cp=True, overflow=False, edges=True):
    '''
    Extension of the hist2array
    '''
    print("Opening",hist_path,"in",file_path)
    tfile=TFile.Open(file_path)
    thist=tfile.Get(hist_path)
    print("Opened", thist, type(thist[0]))

    freq, edges = hist2array(thist, include_overflow=overflow, copy=cp, return_edges=edges)
    #extract the edges, and ensure int frequency count
    edges=edges[0]
    freq=freq.astype(int)

    #find number of bins
    n_bins=len(freq)
    #find the bin width 
    dBin=edges[1]-edges[0]
    #array of bin centres
    binC=np.linspace(edges[0]+dBin/2, edges[-1]-dBin/2, n_bins)
    #print(binC)
    #full histo data
    data=[]
    # add bin centre f times 
    for i_bin, f in enumerate(freq):
        for i in range(f): 
            data.append(binC[i_bin])

    return data, n_bins