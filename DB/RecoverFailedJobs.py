############
## Assess which runs are missing 
############
import argparse, sys, glob, os # glob is awesome! allows regex in process calls 
import subprocess, shlex 
import re 
from os import listdir
from os.path import isfile, join
import numpy as np
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--source")  # file
parser.add_argument("--output") # file
args = parser.parse_args()
source=str(args.source)
output=str(args.output)

BNL_ctag=8.6348e9 

#store the unique ids (run)
source_token = [] 
output_token = []

ctags=[]
run_types=[]

print("Source file:", source)
f_source = open(source, "r")
count_line = len(f_source.readlines(  ))
print("Total of", count_line, "runs")
f_source = open(source, "r")
for i_fileName in f_source:
    fileName_str =  i_fileName.split(',') # remove extension 
    run = int(fileName_str[0])
    run_type = str(fileName_str[1])
    ctag = float(fileName_str[5])
    source_token.append(run)
    ctags.append(ctag)
    run_types.append(run_type)

print("Extracted", len(source_token), "tokens of form 'run.subrun', e.g.", source_token[0])
# print ctags
# print run_types
    
print("Output file:", output)
f_source = open(output, "r")
count_line = len(f_source.readlines(  ))
print("Total of", count_line, "runs")
f_source = open(output, "r")
for i_fileName in f_source:
    fileName_str =  i_fileName.split(' ') # remove extension 
    run = int(fileName_str[0])
    output_token.append(run)

print("Number of files in source-output", len(source_token)-len(output_token))

#Some sanity checks 
print("Double counted files (should be empty []:)")
print [item for item, count in collections.Counter(source_token).items() if count > 1]

print("Computing list of missing source tokens...")
main_list = [x for x in source_token if x not in output_token]
print("Number of found differences (runs in source that are not in output)", len(main_list))
print(main_list)

print("Computing list of missing output tokens...")
main_list_2 = [x for x in output_token if x not in source_token]
print("Number of found differences (runs in output that are not in source)", len(main_list_2))
print(main_list_2)


#now form and use maps to get ctags from missing runs



'''
print("Computing list of missing source file paths...")
#from new list for input paths to next processes 
input_list = []
f_source = open(source, "r")
for i_fileName in f_source:
    for i_missing in main_list:
        fileName_str =  i_fileName.split('.') # remove extension 
        run = fileName_str[0].split("_")[2] # run is in the 2nd part "_"
        run= int(re.search(r'\d+', run).group()) # remove "run" string 
        subrun = fileName_str[0].split("_")[3] # subrun is in the 3rd part "_"
        token = str(run)+"."+str(subrun)
        if (i_missing == token):
            input_list.append(i_fileName)
            
f_source.close()
print("New list contains ", len(input_list))
print(input_list)

f=open("FileList_recovery.txt", "w+")
for i_file in input_list:
    f.write(str(i_file))        

f.close()
print('Recovery file written for grid submission: FileList_recovery.txt')
'''
