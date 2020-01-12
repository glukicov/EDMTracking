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
    Extension of the hist2array to return 1D histo data as array 
    Returns dataX[], n_binsX, dBinX
    len(data) = entries in histogram (not counting over/underflows))
    Works for 1D (TH1) ROOT  histograms 

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
    
    D=len(freq.shape)
    if (D != 1):
        raise Exception("Implementation for 1D got dimensions of", D, "use hist2D2np() method instead!")
        
    #extract the edges from the array of arrays
    # ensure int frequency count
    edges=edges[0]
    freq=freq.astype(int)

    #find number of bins
    n_bins=len(freq)
    
    #find the bin width 
    dBin=edges[1]-edges[0]

    #array of bin centres
    binC=np.linspace(edges[0]+dBin/2, edges[-1]-dBin/2, n_bins)
     
    #expand frequencies by bin centres 
    #bin with 0 frequency do not contribute to this expansion
    data=[]
    for i_bin, f in enumerate(freq):
        for i in range(0, f): 
            data.append(binC[i_bin])

    if (len(data) != exp_total):
        raise Exception("Did not get expected entries! Got", len(data), "expected", exp_total)
        sys.exit()

    return data, n_bins, dBin

def hist2D2np(file_path="data/data.root", hist_path="Tracks/pvalue", cp=True, overflow=False, edges=True):
    '''
    Extension of the hist2array to return histo data as array 
    Returns dataXY[D], n_binsXY[D], dBinXY[D] with dimensions equal to dimension (D=2) of the input histogram
    len(data[D]) = entries in histogram (not counting over/underflows))
    Work for 2D histograms (TH2)

    #Example of getting some data, bins and bind width from ROOT 2D Histogram
    # dataXY, binsXY, dBinXY = ru.hist2D2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/radialPos")
    '''
    print("Opening",hist_path,"in",file_path)
    tfile=TFile.Open(file_path)
    thist=tfile.Get(hist_path)
    exp_total = int(thist.Integral()) # total number of entries (not counting over/underflows)
    print("Opened", thist, type(thist[0]), "with", exp_total, "entries (exc. over/underflows)")

    #now call the the default function to get frequencies and bin edges 
    freq, edges = hist2array(thist, include_overflow=overflow, copy=cp, return_edges=edges)
    
    D=len(freq.shape)
    print("Dimension:", D)
    if (D != 2):
        raise Exception("Implementation 2D, got dimensions of", D, "use hist2np() method instead!")
    # prepare storage based on D dimensions
    data=[[], []] 
    n_bins=[]
    dBin=[] 
    binC=[] # X, Y values 

    #ensure int frequency count
    freq=freq.astype(int)
    
    # loop over dimensions and fill bin centres
    for i_dim in range(D):
       
        #extract the edges from the array of arrays
        edges_D=edges[i_dim]
       
        #find number of bins
        n_bins_D=len(edges_D)-1 # 1 more edge than bin centres 
        n_bins.append(n_bins_D)
    
        #find the bin width 
        dBin_D=edges_D[1]-edges_D[0]
        dBin.append(dBin_D)
        
        #array of bin centres
        binC_D=np.linspace(edges_D[0]+dBin_D/2, edges_D[-1]-dBin_D/2, n_bins_D)
        binC.append(binC_D)
        
    # now correlate binC(x,y) and freq(x,y)
    for ix,iy in np.ndindex(freq.shape):
        for i in range(0, freq[ix][iy]): # for that number of freq 
                data[0].append(binC[0][ix]) #append to  X, and Y
                data[1].append(binC[1][iy])
            
    # sanity check
    for i_dim in range(D):
        if (len(data[i_dim]) != exp_total):
            raise Exception("Did not get expected entries! Got ", len(data[i_dim]), "for D:", i_dim, "expected", exp_total)
            sys.exit()

    return data, n_bins, dBin