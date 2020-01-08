# Author: Gleb Lukicov (7 Jan 2020)
# Based on Saskia's ROOT macro 
# Test the "injection" EDM blinding

#Blinding lib
import os, sys
sys.path.append(os.environ['Blind_Path']) # path to Blinding libs from profile 
from BlindersPy3 import Blinders
from BlindersPy3 import FitType

#other imports
import argparse
import numpy as np
import matplotlib.pyplot as plt

# import faulthandler; faulthandler.enable()

#Input constants
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--n_inject", type=int, default=100)
arg_parser.add_argument("--d0", type=float, default=1.9e-19) # BNL edm limit in e.cm
arg_parser.add_argument("--ppm", type=float, default=1e-6)
# blinding constructor constants 
arg_parser.add_argument("--R", type=float, default=3.5) #  nominal R
arg_parser.add_argument("--boxWidth", type=float, default=0.3)
arg_parser.add_argument("--gausWidth", type=float, default=0.8)
args=arg_parser.parse_args()
n_inject=args.n_inject
d0=args.d0
ppm=args.ppm
R=args.R
boxWidth=args.boxWidth
gausWidth=args.gausWidth

#edm constants
e = 1.6e-19 # J
aMu = 11659208.9e-10 
mMu = 105.6583715 # u
mMuKg = mMu * 1.79e-30 # kg
B = 1.451269 # T
c = 299792458. # m/s
cm2m = 100.0
hbar = 1.05457e-34
pmagic = mMu/np.sqrt(aMu)
gmagic = np.sqrt( 1.+1./aMu )
beta   = np.sqrt( 1.-1./(gmagic*gmagic) )

#storage for plotting
idiff_array = np.array([])

def main():
    data = test_blidning(n_inject)
    # plot(data)

def test_blidning(n_inject):
    print("R:", R, "boxWidth:", boxWidth, "gausWidth:", gausWidth) 

    # we will append the iterator to make a blinding phrase
    # different for each injection
    for injection in range(n_inject):
        blinding_string = "There we go "+str(injection)
        # apply Omega_a blinding ("EDM style") with the blinding_string and input pars 
        print(blinding_string)
        getBlinded = Blinders(FitType.Omega_a, blinding_string, boxWidth, gausWidth)

        # use the blinder to give us a blinded offset from the input R which is 10*d0
        iAmp = getBlinded.paramToFreq(R) # this is something like 1.442 which is the shifted / blinded omegaA value
        iRef  = getBlinded.referenceValue() # this is 1.443 which is the reference omegaA value
        iDiff =  ((iAmp / iRef) - 1) / ppm # this is (R - Rref) in units of ppm
        print("iDiff:", iDiff, "iRef:", iRef, "iAmp:", iAmp)

#         # iDiff tells us the edm value we have got e.g. input EDM_blind = iDiff * d0
#         EDM_blind = iDiff * d0
#         print("EDM_blind:", EDM_blind)

#         omegaA = iRef # just use the reference value

#         eta_blind = ((4 * mMuKg * c * EDM_blind)/ (hbar * cm2m) )
#         tan_delta_blind = (eta_blind * beta) / (2 * aMu)
#         delta_blind = atan(tan_delta_blind)
    
#         print("aMu:", aMu, "beta:", beta, "delta_blind:", delta_blind)
        
#         idiff_array.append(iDiff)

#         return idiff_array

# def plot(data):
#     pass

if __name__ == "__main__":
    main()