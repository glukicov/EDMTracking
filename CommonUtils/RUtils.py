# Some common tools to go from ROOT Histogram to array-type structure 
# Gleb Lukicov (11 Jan 2020)  

# Based on root_numpy package: 
# http://scikit-hep.org/root_numpy/
# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.hist2array.html

from ROOT import TH1, TH2, TFile
import numpy as np
from root_numpy import hist2array
import sys

def hist2np(file_path="data/data.root", hist_path="Tracks/pvalue", cp=True, overflow=False, edges=True):
    '''
    Extension of the hist2array to return full histo data as array 
    Returns data[D], n_bins[D] with dimensions equal to dimension (D) of the input histogram
    len(data[D]) = entries in histogram (not counting over/underflows))

    #Example of getting some data, bins and bind width from ROOT 1D Histogram
    # dataX, binsX, dBinX = ru.hist2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/radialPos")
    '''
    print("Opening",hist_path,"in",file_path)
    tfile=TFile.Open(file_path)
    thist=tfile.Get(hist_path)
    exp_total = int(thist.Integral()) # total number of entries (not counting over/underflows)
    print("Opened", thist, type(thist[0]), "with", exp_total, "entries (exc. over/underflows)")

    #now call the the default function to get frequencies and bin edges 
    freq, edges = hist2array(thist, include_overflow=overflow, copy=cp, return_edges=edges)
    
    print("edges", edges)
    print("len(edges)", len(edges))
    print("freq", freq)
    print("len(freq)", len(freq))

    print("freq.shape", freq.shape)
    print("edges.shape", edges.shape)
    
    D=len(freq.shape)
    print("Dimension:", D)
    # prepare storage based on D dimensions
    data=[]
    n_bins=[]
    dBin=[] 
    
    # loop over dimensions and fill storage arrays 
    for i_dim in range(D):
        
        #extract the edges, and ensure int frequency count
        edges=edges[0]
        freq=freq.astype(int)

        # tmp storage array for this dimension
        data_D=[]
        #find number of bins
        n_bins_D=len(freq)
        print("n_bins",n_bins)
        #find the bin width 
        dBin_D=edges[1]-edges[0]
        print("dBin", dBin)

        print("edges", edges)
        print("len(edges)", len(edges))
        print("freq", freq)
        print("len(freq)", len(freq))

        #array of bin centres
        binC=np.linspace(edges[0]+dBin/2, edges[-1]-dBin/2, n_bins)
        print("binC", binC)
        print("len(binC)", len(binC))
        
        # add bin centre f times 
        count = 0 
        countZeroes = len([i for i in freq if i == 0]) 
        print("Zero bins:", countZeroes)
        print("non-zero bins:", n_bins-countZeroes)

        #bin with 0 frequency do not contribute to this expansion
        for i_bin, f in enumerate(freq):
            # print(i_bin, f)
            for i in range(0, f): 
                data.append(binC[i_bin])

        print("len(data)", len(data))
        if (len(data) != exp_total):
            raise Exception("Did not get expected entries! Got", len(data), "expected", exp_total)
            sys.exit()

    return data, n_bins, dBin