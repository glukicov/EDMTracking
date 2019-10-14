#!/bin/bash

####Test script to submit test jobs to the UK grid###

# gm2 version constants and user constants
export VERSION=v9_29_00
export FLAG=prof
export USER=glukicov # user specific 
export devArea=gm2Soft  # user specific  
echo "Using gm2 ${VERSION} with flag ${FLAG} as user ${USER} with devArea ${devArea}"
GROUP=$EXPERIMENT
echo "Group: ${GROUP}"

#Environment printout 
echo "Start: "`date`
echo "Site:${GLIDEIN_ResourceName}"
echo "the worker node is " `hostname` "OS: " `uname -a`
echo "the user id is " `whoami`
echo "the output of id is " `id`
echo "the process id is ${PROCESS}" # unique for each job, used to get back the output file and log

echo "Here is the your environment in this job: " > job_output_${CLUSTER}.${PROCESS}.log
env >> job_output_${CLUSTER}.${PROCESS}.log

#################################
# set up common software environments, if necessary 
#################################
echo "setup cvmfs common products"
source /cvmfs/fermilab.opensciencegrid.org/products/common/etc/setups # ifdh tool to copy files
setup ifdhc -z /cvmfs/fermilab.opensciencegrid.org/products/common/db # ifdh tool to copy files
source /cvmfs/fermilab.opensciencegrid.org/products/larsoft/setup  # might be useful 
# setup sam_web_client  # might be useful for SAM DS management 
echo "done setting up cvmfs common products "

####################################################
# setup the g-2 software
####################################################
#echo "Setting up g-2 software"
source /cvmfs/gm2.opensciencegrid.org/prod/g-2/setup
echo "Setting up: /cvmfs/gm2.opensciencegrid.org/prod/g-2/setup"
setup gm2 ${VERSION} -q ${FLAG}
echo "Setting up: setup gm2 ${VERSION} -q ${FLAG}"
echo "Done setting up g-2 software"

# For this simple test, run a MC for 1 event: MDC1 comes directly from CVMFS
echo "Running MDC1 for 1 event"
gm2 -c mdc1.fcl -n 1
echo "Finished running MDC1 for 1 event"

echo "Job finished successfully on: " `date`  

exit 0