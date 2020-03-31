# Author: Gleb Lukicov (21 Feb 2020)
# A front-end module to run over fitWithBlinders_skim.py iteratively  

import sys, re, os, subprocess, datetime, glob
import time as timeMod
import argparse

#Input fitting parameters 
arg_parser = argparse.ArgumentParser()
args=arg_parser.parse_args()


def main():
    '''
    As a quick solution use sub process 
    '''

    # keys=["bin", "period", "lt", "p_min", "p_both", "phase", "start", "stop"]
    # keys_dirs=["bin_w", "g2period", "lt", "p_min", "p_both", "phase", "start", "stop"]
    keys=["bin_w", "g2period"]
        
    for key in keys:
        # subprocess.call(["python3", "Scans_edm.py", "--"+key])
        # subprocess.call(["python3", "Scans_edm.py", "--plot_"+key])
        os.chdir("../fig/scans_fom/"+key)
        subprocess.call(["../../../docs/scan.sh"], shell=True)


if __name__ == "__main__":
    main()