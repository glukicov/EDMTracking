#!/bin/bash
set -x
echo Start  `date`
echo Site:${GLIDEIN_ResourceName}
echo "the worker node is " `hostname` "OS: "  `uname -a`
echo "You are running as user `whoami`"

#Always cd to the scratch area!

cd $_CONDOR_SCRATCH_DIR

IFDH_OPTION=""

# set group based on the EXPERIMENT evnironment variable set by jobsub
GROUP=$EXPERIMENT

if [ -z $GROUP ]; then

# try to figure out what group the user is in
GROUP=`id -gn`
fi


case $GROUP in

e938)
SCRATCH_DIR=/pnfs/minerva/scratch/users
;;
minerva)
SCRATCH_DIR=/pnfs/minerva/scratch/users
;;
e875)
#need to check this
SCRATCH_DIR=/pnfs/minos/scratch/users
;;
minos)
#need to check this                                                                                                                                                                                                                           
SCRATCH_DIR=/pnfs/minos/scratch/users
;;
mars)
SCRATCH_DIR=""
;;
lbnemars)
SCRATCH_DIR="/lbne/data/lbnemars/users/"
;;
marslbne)
SCRATCH_DIR="/lbne/data/marslbne/users/"
;;
marsmu2e)
SCRATCH_DIR=""
;;
marsgm2)
SCRATCH_DIR=""
;;
marsaccel)
SCRATCH_DIR=""
;;
larrand)
#pnfs/scene? probably not....
SCRATCH_DIR=""
;;
nova)
SCRATCH_DIR=/pnfs/nova/scratch/users
;;
t-962)
SCRATCH_DIR="/argoneut/data/users"
;;
argoneut)
SCRATCH_DIR="/argoneut/data/users"
;;
mu2e)
SCRATCH_DIR=/pnfs/mu2e/scratch/users
;;
microboone)
SCRATCH_DIR=/pnfs/uboone/scratch/users
;;
uboone)
SCRATCH_DIR=/pnfs/uboone/scratch/users
;;
lbne)
SCRATCH_DIR=/pnfs/lbne/scratch/users
;;
seaquest)
SCRATCH_DIR="/e906/data/users/condor-tmp"
IFDH_OPTION="--force=cpn"
;;
e906)
SCRATCH_DIR="/e906/data/users/condor-tmp"
IFDH_OPTION="--force=cpn"
;;
coupp)
SCRATCH_DIR=""
;;
gm2)
# g-2 does not allow the gm2ana user to write to pnfs so we have to use blueArc for now
#SCRATCH_DIR=/gm2/data/users
SCRATCH_DIR=/pnfs/GM2/scratch/users
;; 
t-1034)
# lariat... no pnfs yet
SCRATCH_DIR=/pnfs/lariat/scratch/users
;;
lariat)
SCRATCH_DIR=/pnfs/lariat/scratch/users
;;
darkside)
SCRATCH_DIR="/pnfs/darkside/scratch/users"
;;
lar1nd)
SCRATCH_DIR="/pnfs/lar1nd/scratch/users"
;;
lsst)
SCRATCH_DIR="/pnfs/lsst/scratch/users"
;;
annie)
SCRATCH_DIR=""
;;
numix)
SCRATCH_DIR=""
;;
fermilab)
SCRATCH_DIR="/pnfs/fermilab/volatile"
#SCRATCH_DIR="/grid/data"
export CPN_LOCK_GROUP=gpcf
;;
esac

voms-proxy-info --all

### Force use of SLF6 versions for systems with 3.x kernels
case `uname -r` in
    3.*) export UPS_OVERRIDE="-h Linux64bit+2.6-2.12";;
esac

#source /cvmfs/oasis.opensciencegrid.org/fermilab/products/common/etc/setup
#source /cvmfs/oasis.opensciencegrid.org/fermilab/products/larsoft/setup
#setup ifdhc v1_5_1a -z /cvmfs/oasis.opensciencegrid.org/fermilab/products/common/db

source /cvmfs/fermilab.opensciencegrid.org/products/common/etc/setups
source /cvmfs/fermilab.opensciencegrid.org/products/larsoft/setup
setup ifdhc v1_6_2 -z /cvmfs/fermilab.opensciencegrid.org/products/common/db

echo "Here is the your environment in this job: " > job_output_${CLUSTER}.${PROCESS}.log
env >> job_output_${CLUSTER}.${PROCESS}.log

echo "group = $GROUP"

# If GRID_USER is not set for some reason, try to get it from the proxy
if [ -z ${GRID_USER} ]; then
GRID_USER=`basename $X509_USER_PROXY | cut -d "_" -f 2`
fi

echo "GRID_USER = `echo $GRID_USER`"

sleep $[ ( $RANDOM % 10 )  + 1 ]m

umask 002

if [ -z "$SCRATCH_DIR" ]; then
    echo "Invalid scratch directory, not copying back"
    echo "I am going to dump the log file to the main job stdout in this case."
    cat job_output_${CLUSTER}.${PROCESS}.log
else

# Very useful for debugging problems with copies
export IFDH_DEBUG=1

    # first do lfdh ls to check if directory exists
    ifdh ls ${SCRATCH_DIR}/$GRID_USER
    # A non-zero exit value probably means it doesn't exist yet, or does not have group write permission, 
    # so send a useful message saying that is probably the issue
    if [ $? -ne 0 && -z "$IFDH_OPTION" ] ; then	
	echo "Unable to read ${SCRATCH_DIR}/$GRID_USER. Make sure that you have created this directory and given it group write permission (chmod g+w ${SCRATCH_DIR}/$GRID_USER)."
	exit 74
    else
        # directory already exists, so let's copy
	ifdh cp -D $IFDH_OPTION job_output_${CLUSTER}.${PROCESS}.log ${SCRATCH_DIR}/${GRID_USER}
	if [ $? -ne 0 ]; then
	    echo "Error $? when copying to dCache scratch area!"
	    echo "If you created ${SCRATCH_DIR}/${GRID_USER} yourself,"
	    echo " make sure that it has group write permission."
	    exit 73
	fi
    fi
fi

echo "End `date`"

exit 0
