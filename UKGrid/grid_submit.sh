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

#### Copy local products from PNFS to the grid 
# XXX Can PNFS be seen from Liverpool? 
# XXX We might not need to do that, as all the tracking is done from the default modules in develop branch on cvmfs

# XXX For this simple test, avoid copying 
# cd $_CONDOR_SCRATCH_DIR # good practice on OSG 
# echo "copying over /pnfs/gm2/scratch/users/${USER}/localArea/gm2_${VERSION}_${USER}_gm2Dev_${VERSION}.tgz"
# ifdh cp /pnfs/gm2/scratch/users/${USER}/localArea/gm2_${VERSION}_${USER}_gm2Dev_${VERSION}.tgz ./gm2_${VERSION}_${USER}_gm2Dev_${VERSION}.tgz
# echo "ls:" `ls`
# echo "Unzipping the local area"
# tar -xvzf gm2_${VERSION}_${USER}_gm2Dev_${VERSION}.tgz
# echo "Unzipped the local area"   
# echo "ls:" `ls`

###Copy data file to the grid
# XXX Can PNFS be seen from Liverpool? 

# XXX For this simple test, avoid copying 
# ifdh cp /pnfs/gm2/scratch/users/${USER}/testGrid/gm2tracker_unpack_22703105_26067.00063.root gm2tracker_unpack_22703105_26067.00063.root

###Run the gm2 command 
# cd gm2/app/users/${USER}/${devArea}/gm2Dev_${VERSION}/localProducts_gm2_${VERSION}_${FLAG}/gm2tracker/ 
# gm2 -c test.fcl -S FileList.txt

# XXX For this simple test, run a MC for 1 event (MC is already on CVMFS, my TestNoOutput_module.cc is not until next release)
# MDC1 comes directly from CVMFS
gm2 -c mdc1.fcl -n 1

### Copy the output job data and log files back 
# XXX For this simple test, avoid copying 
# ifdh cp test.root /pnfs/GM2/scratch/users/${USER}/testGrid/test_${PROCESS}.root
# ifdh cp test.log /pnfs/GM2/scratch/users/${USER}/testGrid/test_${PROCESS}.log

echo "Job finished successfully on: " `date`  

###Get the job log 
#jobsub_fetchlog -G gm2 --jobid=

exit 0