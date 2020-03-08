'''
Some common tools to go from ROOT Histogram to array-type structure 
Gleb Lukicov (11 Jan 2020)  

(!!!) This is Cython module, and needs to be compiled to be used with Python
 python3 setup.py build_ext --inplace
the subsequent import into Python code/Jupyter cell is just as for a normal Py module
e.g. 

   import sys
   sys.path.append("../CommonUtils/")
   import RUtils as ru
'''

from ROOT import TH1, TH2, TFile
import numpy as np
from root_numpy import hist2array # http://scikit-hep.org/root_numpy/
cimport cython  

def hist2np(str file_path="data/data.root", str hist_path="Tracks/pvalue", bint cp=True,  bint overflow=False,  bint edges_bool=True):
    '''
    Extension of the hist2array to return 1D or 2D histos data as an array 
    Returns: 1D: dataX[], n_binsX, dBinX  2D: dataXY[][], n_binsXY[], dBinXY[]
    len(dataX) = entries in histogram (not counting over/underflows))
    (!) Works for non-normalised (to 1) histos only (can be implemented) 

    #Example of getting some data, bins and bind width from ROOT 1D or 2D Histogram
    # dataXY, binsXY, dBinXY = ru.hist2np(file_path="DATA/noEDM.root", hist_path="AllStationsNoTQ/VertexExtap/t>0/0<p<3600/radialPos")
    '''
    print("RUtils::hist2np Opening",hist_path,"in",file_path)
    tfile=TFile.Open(file_path)
    thist=tfile.Get(hist_path)
    cdef int exp_total = int(thist.Integral()) # total number of entries (not counting over/underflows)
    print("RUtils::hist2np Opened", thist, type(thist[0]), "with", exp_total, "entries (exc. over/underflows)")

    #now call the the default root_numpy function to get frequencies and bin edges 
    # # http://scikit-hep.org/root_numpy/reference/generated/root_numpy.hist2array.html
    freq, edges = hist2array(thist, include_overflow=overflow, copy=cp, return_edges=edges_bool)
    
    cdef int D=len(freq.shape)
    if(D!=1 and D!=2):
        raise Exception("RUtils::hist2np Implementation for 1D or 2D, got dimensions of", D)

    #ensure int frequency count (in place copy)
    freq=freq.astype(int, copy=False)
    
    if (D == 1):
        
        #extract the edges from the array of arrays
        edges=edges[0]

        #find number of bins
        n_bins=len(edges)-1 # 1 more edge than bin centres
        
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
            raise Exception("RUtils::hist2np Did not get expected entries! Got", len(data), "expected", exp_total)
    #D1 end
    
    # TH2 
    if (D == 2):
        
        # prepare storage based on D dimensions (X,Y) 
        data=[[], []]  
        # X, Y values will be appended  
        n_bins=[]
        dBin=[] 
        binC=[] 
        
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
        for ix,iy in np.ndindex(freq.shape): # looping over x,y indices of a matrix 
            for i in range(0, freq[ix][iy]): # for that number of freq 
                    data[0].append(binC[0][ix]) #append to  X, and Y
                    data[1].append(binC[1][iy])
                
        # sanity check
        for i_dim in range(D):
            if (len(data[i_dim]) != exp_total):
                raise Exception("Did not get expected entries! Got ", len(data[i_dim]), "for D:", i_dim, "expected", exp_total)            
    # D2 end

    return np.array(data), n_bins, dBin