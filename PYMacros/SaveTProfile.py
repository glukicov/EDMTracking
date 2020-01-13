from ROOT import TFile, TH2, TProfile

import os, sys
import argparse 

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--file_path", type=str, default="DATA/VLEDM.root") 
arg_parser.add_argument("--hist_path", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
arg_parser.add_argument("--bins", type=str, default="AllStationsNoTQ/VertexExt/t>0/0<p<3600/thetay_vs_time_modg2") 
args=arg_parser.parse_args()

#open file and histo 
tfile=TFile.Open(args.file_path)
thist=tfile.Get(args.hist_path)

#