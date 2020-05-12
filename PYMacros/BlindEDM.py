# Author: Gleb Lukicov (7 Jan 2020)
# Based on Saskia's ROOT macro 
# Use the "injection" EDM blinding - for "delta" (A_EDM measured by the trackers)

import os, sys
import math
import numpy as np
#Blinding lib
sys.path.append(os.environ['Blind_Path']) # path to Blinding libs from profile 
from BlindersPy3 import Blinders
from BlindersPy3 import FitType

#Input constants
d0=1.9e-19 # BNL edm limit in e.cm
ppm=1e-6 
R=100
boxWidth=0.3
gausWidth=0.8

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

def main():
    delta_blind = get_delta_blind(boxWidth=boxWidth, gausWidth=gausWidth, R=R)
    
def get_delta_blind():      
    # apply Omega_a blinding ("EDM style") with the blinding_string and input pars 
    edmBlinded = Blinders(FitType.Omega_a, "EDM all day", boxWidth, gausWidth)

    # use the blinder to give us a blinded offset from the input R which is 10*d0
    iAmp = edmBlinded.paramToFreq(R) # this is something like 1.442 which is the shifted / blinded omegaA value
    iRef  = edmBlinded.referenceValue() # this is 1.443 which is the reference omegaA value
    iDiff =  ((iAmp / iRef) - 1) / ppm # this is (R - Rref) in units of ppm

    # iDiff tells us the edm value we have got e.g. input EDM_blind = iDiff * d0
    EDM_blind = iDiff * d0

    # convert to physics
    eta_blind = ((4 * mMuKg * c * EDM_blind)/ (hbar * cm2m) )
    tan_delta_blind = (eta_blind * beta) / (2 * aMu)
    delta_blind = math.atan(tan_delta_blind)

    return delta_blind
    
if __name__ == "__main__":
    main()