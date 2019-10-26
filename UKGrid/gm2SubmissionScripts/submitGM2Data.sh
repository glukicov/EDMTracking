#This script is NOT supposed to be run interactively
#It is a part of production submission script
##################################################################


###############################################
# set the base URI for our samweb service
###############################################
export IFDH_BASE_URI="http://samweb.fnal.gov:8480/sam/gm2/api"
export IFDH_DEBUG=0
export IFDH_CP_MAXRETRIES=2

###############################################
# set the environment for Xrootd
###############################################
export XRD_REQUESTTIMEOUT=14400
export XRD_CONNECTIONRETRY=32

######################################################
# If we don't have the g-2 CVMFS, then bail
######################################################	
if [ ! -d /cvmfs/gm2.opensciencegrid.org ]; then
  echo “CVMFS gm2 repo seems to not be present. Sleeping and then exiting.”
  sleep 100
  exit 1
fi

#############################
# environment variables
#############################

RMAINFCLNAME="None"
RFCLNAMELIST=()
NEVTSPERJOB=-1
STARTEVENT=1
USE_LOCAL_SETUP=0
INPUT_RELEASE=""

# multiple fcls 
MULTIPLE_FCL="NO"

# useful when running with localArea
# can look this up by "ups active" from a local area
QUAL="e14:prof"


#############################
# run conditions
#############################

DATA_TYPE="None"

INCLUDE_ANA=0

RUN=1
SUBRUN=1
REQUEST=9999
MEMORY_TRACKER=0
VERBOSE=0
TRACE=""
TIMING=""

TAG="offline"
TAGTMP="None"
SYSTEM="None"
KICKER=1
QUAD=1
CONFIG="nominal"

#One output file (set) per input file?
ONE_BY_ONE="NO"

#single root file? (one output set could already contain multiple root files)
MULTIPLE_ROOT="NO"

OUTPUT_ROOT=""
OUTPUT_DIR=""

#copy json file?
JSON=0


#############################
# exit status
#############################
STATUS=0

# set exit status to be the last non-zero status (if there is any non-zero)
set -o pipefail

##########################################
# special strings
##########################################

message_categories="@local::message_categories"
kickerpath="auxChain.synckicker" 
quadpath="auxChain.quad"

##########################################
# specific affinity option
# by default no affinity
##########################################

CORE="None" 
TASKSET=""

##########################################
# specific simulation environment variables
##########################################
PARTICLE="muon"
GUN="None"
MUONSPERFILL=1

##########################################
# specific SAM environment variables
##########################################
HOST=`/bin/hostname`

############################################################################
# get arguments TODO figure out more efficient way to match variables
############################################################################
nargs=$#

#echo "nargs=$nargs"

for (( i=1; i<=nargs; ++i ))
do
   key="$1"
   value="$2"
   
   #echo "key=$key"
   #echo "value=$value"

   if [[ $key == "remoteFile" ]] ; then
      RMAINFCLNAME="$value"
   fi

   if [[ $key == "remoteFiles" ]] ; then
      RFCLNAME="$value"
   fi

   if [[ $key == "localArea" && $value == "True" ]] ; then
      USE_LOCAL_SETUP=1
   fi

   if [[ $key == "release" ]] ; then
      INPUT_RELEASE="$value"
   fi

   if [[ $key == "qual" ]] ; then
      QUAL="$value"
   fi

   if [[ $key == "first-event" ]] ; then
      STARTEVENT=$value
   fi

   if [[ $key == "nevents" ]] ; then
      NEVTSPERJOB=$value
   fi

   if [[ $key == "outfiles" ]] ; then
      OUTPUT_DIR="$value"
   fi

   if [[ $key == "onebyone" && $value == "True" ]] ; then
      ONE_BY_ONE="YES"
   fi

   if [[ $key == "multipleroot" && $value == "True" ]] ; then
      MULTIPLE_ROOT="YES"
   fi

   if [[ $key == "runNumber" ]] ; then
      RUN="$value"
   fi

   if [[ $key == "requestid" ]] ; then
      REQUEST="$value"
   fi

   if [[ $key == "config" ]] ; then
      CONFIG="$value"
   fi

   if [[ $key == "nmuonsPerFill" ]] ; then
      MUONSPERFILL="$value"
   fi

   if [[ $key == "memorytracker" && $value == "True" ]] ; then
      MEMORY_TRACKER=1
   fi

   if [[ $key == "verbose" && $value == "True" ]] ; then
      VERBOSE=1
   fi

   if [[ $key == "trace" && $value == "True" ]] ; then
      TRACE="--trace"
   fi

   if [[ $key == "timing" && $value == "True" ]] ; then
      TIMING="--timing"
   fi

   if [[ $key == "core" ]] ; then
      CORE=$value
   fi

   #get data type
   if [[ $key == "mc" && $value == "True" ]] ; then
      DATA_TYPE="$key"
   elif [[ $key == "daq" && $value == "True" ]] ; then
      DATA_TYPE="$key"
   fi


   #get ana tier
   if [[ $key == "ana" && $value == "True" ]] ; then
      INCLUDE_ANA=1
   fi


   #get particle gun
   if [[ $key == "beamGun" && $value == "True" ]] ; then
      GUN="beamgun"
   elif [[ $key == "gasGun" && $value == "True" ]] ; then
      GUN="gasgun"
   elif [[ $key == "miGun" && $value == "True" ]] ; then
      GUN="migun"
   elif [[ $key == "simpleGun" && $value == "True" ]] ; then
      GUN="simplegun"
   fi


   #get particle type
   if [[ $key == "muon" && $value == "True" ]] ; then
      PARTICLE="$key"
   elif [[ $key == "proton" && $value == "True" ]] ; then
      PARTICLE="$key"
   elif [[ $key == "pion" && $value == "True" ]] ; then
      PARTICLE="$key"
   fi


   #get system type
   if [[ $key == "ring" && $value == "True" ]] ; then
      SYSTEM="ring"
      TAG="ringsim"
   elif [[ $key == "harp" && $value == "True" ]] ; then
      SYSTEM="$key"
      TAG="fiberharp"
   elif [[ $key == "tracker" && $value == "True" ]] ; then
      SYSTEM="$key"
      TAG="tracker"
   elif [[ $key == "calo" && $value == "True" ]] ; then
      SYSTEM="$key"
      TAG="calo"
   elif [[ $key == "kickeroff" && $value == "True" ]] ; then
      KICKER=0
   elif [[ $key == "quadoff" && $value == "True" ]] ; then
      QUAD=0
   fi

   # get TAG info
   if [[ $key == "tag" && $value != "None" ]] ; then
      TAGTMP="$value"
   fi

   #get json information
   if [[ $key == "json" && $value == "True" ]] ; then
      JSON=1
   fi


   shift

done


#overwrite parameters
if [[ ${TIER} == "truth" && ${SYSTEM} == "harp" ]] ; then
   TAG="ringsim"
fi

if [[ ${TIER} == "full" && ${DATA_TYPE} == "mc" && ${SYSTEM} != "harp" && $SYSTEM != "calo" && $SYSTEM != "tracker" ]] ; then
   TAG="ringsim"
fi

if [[ "${GUN}" == "None" ]] ; then
   if [[ "${RMAINFCLNAME}" == *Beam* ]] ; then
      GUN="beamgun"
   elif [[ "${RMAINFCLNAME}" == *Gas* ]] ; then
      GUN="gasgun"
   elif [[ $INPUT_FILE == *beam* ]] || [[ $INPUT_FILE == *Beam* ]] || [[ $INPUT_FILELIST == *beam* ]] || [[ $INPUT_FILELIST == *Beam* ]]; then
      GUN="beamgun"
   elif [[ $INPUT_FILE == *gas* ]] || [[ $INPUT_FILE == *Gas* ]] || [[ $INPUT_FILELIST == *gas* ]] || [[ $INPUT_FILELIST == *Gas* ]]; then
      GUN="gasgun"
   fi
fi

if [[ ${TAGTMP} != "None" ]] ; then
   TAG=${TAGTMP}
fi

[ $GRID -eq 0 ] && OFFSITE=0


# Check SAM_FILE_LIMIT
[[ ! -z ${SAM_FILE_LIMIT} ]] && [[ ${SAM_FILE_LIMIT} -lt 0 ]] && SAM_FILE_LIMIT=0
[[ -z ${SAM_FILE_LIMIT} ]] && SAM_FILE_LIMIT=-1

# check CAMPAIGN
if [[ -z $CAMPAIGN ]]; then
   echo -e "\n!!!Undefined CAMPAIGN!!!\n"
   exit 1
fi

##################################
# check fhicl and input files
##################################
if [[ $RMAINFCLNAME == "None" ]]; then
   echo -e "\n!!!Undefined RMAINFCLNAME!!!\n" 
   exit 1 
fi

if [[ $RFCLNAME == "None" ]] || [[ -z $RFCLNAME ]]; then
   echo -e "\n!!!Undefined RFCLNAME!!!\n" 
   exit 1 
fi

IFS=','
RFCLNAMELIST=($RFCLNAME)
unset IFS

[ ${#RFCLNAMELIST[@]} -gt 1 ] && MULTIPLE_FCL="YES"

for FCLNAME in "${RFCLNAMELIST[@]}"; do
    if [ $GRID -eq 0 ]; then
         if [ ! -f "$FCLNAME" ]; then
              echo -e "\n!!!$FCLNAME NOT found!!!\n"
   	      exit 1
         fi
    else 
         if ! cp -f "${CONDOR_DIR_INPUT}/$FCLNAME" .; then
              echo -e "\n!!!Unable to copy ${CONDOR_DIR_INPUT}/$FCLNAME to $PWD !!!\n"
              exit 1
         fi
    fi
done

if [ $GRID -eq 1 -a ${INPUT_FILE} != "None" ]; then
   INPUT_FILE=${CONDOR_DIR_INPUT}/$(basename ${INPUT_FILE})
   if [ ! -f ${INPUT_FILE} ]; then
   	echo -e "\n!!!${INPUT_FILE} NOT found!!!\n"
   	exit 1
   fi
fi

if [ $GRID -eq 1 -a ${INPUT_FILELIST} != "None" ]; then
   INPUT_FILELIST=${CONDOR_DIR_INPUT}/$(basename ${INPUT_FILELIST})
   if [ ! -f ${INPUT_FILELIST} ]; then
   	echo -e "\n!!!${INPUT_FILELIST} NOT found!!!\n"
   	exit 1
   fi
fi


##################################
# create the environment job file
##################################

if [ $GRID -eq 0 ] ; then
   CLUSTER=0
   PROCESS=0
   RELEASE=${INPUT_RELEASE:-$MRB_PROJECT_VERSION}

   if [[ -z ${RELEASE} ]] ; then
     echo -e "\n!!!When running jobs locally (--noGrid), a release version must be specified, see help menu!!!\n"
     exit 1
   fi
fi

ENVLOG="env_${CLUSTER}.${PROCESS}.log"

echo -e "\n>>>Here is the your environment and a few debug statements in this job:\n" | tee -a $ENVLOG
echo -e ">>>gm2 RELEASE=$RELEASE\n" | tee -a $ENVLOG
date -u | tee -a $ENVLOG

#################################
# set up software environments
#################################

echo -e "\n>>>setup cvmfs common products\n " | tee -a $ENVLOG 
source /cvmfs/fermilab.opensciencegrid.org/products/common/etc/setups

# no longer need to setup ifdhc with the new ifdh_art for v8
if [[ $RELEASE == v6* || $RELEASE == v7* || ${IFDH_ART} -eq 0 ]] ; then
  setup ifdhc -z /cvmfs/fermilab.opensciencegrid.org/products/common/db
fi

setup sam_web_client

###########################################################
# add some information if the job is submitted offsite
#
# taken from Adam's docdb 5437
###########################################################
echo -e ">>>Job started on `uname -a`\n" | tee -a $ENVLOG
if [ $OFFSITE -eq 1 ] ; then
   echo -e ">>>Site is $GLIDEIN_Site\n"     | tee -a $ENVLOG
   
   # Not sure what this does
   /cvmfs/grid.cern.ch/util/cvmfs-uptodate /cvmfs/fermilab.opensciencegrid.org

   # Some OSG nodes have newer kernels
   case `uname -r` in
      3.*) export UPS_OVERRIDE="-H Linux64bit+2.6-2.12";;
      4.*) export UPS_OVERRIDE="-H Linux64bit+2.6-2.12";;
   esac

#####################################################
# Do we need help from nova library_shim? Yes, we do!
#####################################################

   export PRODUCTS=$PRODUCTS:/cvmfs/nova.opensciencegrid.org/externals
   setup library_shim v03.03

#Found Nova shim or not?
   if [ $? -ne 0 ]; then 
       echo -e "\n!!!Library_shim NOT FOUND for site $GLIDEIN_Site!!!\n" | tee -a $ENVLOG
       ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs 
       exit 1
   fi


   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_SHIM_SL6_LIB_PATH

   echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a $ENVLOG
   
   echo -e "\nUPS_OVERRIDE=${UPS_OVERRIDE}\n" | tee -a $ENVLOG

fi


####################################################
# setup the g-2 software
####################################################
date -u | tee -a $ENVLOG
echo -e "\n>>>setup g-2 software and ifdh tools\n" | tee -a $ENVLOG

###############################################################
#Override libcurl and libnss packages from library_shim v03.03 
#to (temporarily) fix a NSS bug announced in late April 2019
#Switching to a curl library built with openssl instead of NSS 
###############################################################
setup curl v7_64_1

# Art service implementation in the fhicl does not use ".user" for g-2 software greater than v7
services="services.user"

if [[ $RELEASE == v6* ]]; then
   source /cvmfs/gm2.opensciencegrid.org/prod6/g-2/setup
   [ ${IFDH_ART} -eq 1 ] && setup ifdh_art v1_15_05 -q e10:prof:s41
elif [[ $RELEASE == v7* ]]; then
   source /cvmfs/gm2.opensciencegrid.org/prod7/g-2/setup
   [ ${IFDH_ART} -eq 1 ] && setup ifdh_art v1_15_05 -q e10:prof:s41
elif [[ $RELEASE == v8* ]]; then
   source /cvmfs/gm2.opensciencegrid.org/prod8/g-2/setup
   [ ${IFDH_ART} -eq 1 ] && setup ifdh_art v2_03_06 -q e14:prof:s58
   services="services"
elif [[ $RELEASE == v9* ]]; then
   source /cvmfs/gm2.opensciencegrid.org/prod/g-2/setup # <-- change to prod9 when v10 comes out
   services="services"
   [ ${IFDH_ART} -eq 1 ] && setup ifdh_art v2_05_05 -q e15:prof:s65

else
   echo -e "\n!!!${BASH_SOURCE} needs to be updated for the requested Offline version!!!" | tee -a $ENVLOG
   echo "Requested version is RELEASE='${RELEASE}'" | tee -a $ENVLOG
   exit 1
fi

echo setup gm2 ${RELEASE} -q prof | tee -a $ENVLOG
setup gm2 ${RELEASE} -q prof

if ! which ifdh | tee -a $ENVLOG; then
   echo -e "\n!!!ifdh NOT FOUND!!!\n" | tee -a $ENVLOG
   ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs 
   exit 1
fi

echo | tee -a $ENVLOG

date -u | tee -a $ENVLOG

echo -e "\nCLUSTER=$CLUSTER" | tee -a $ENVLOG
echo "PROCESS=$PROCESS" | tee -a $ENVLOG
echo "DAGMANJOBID=$DAGMANJOBID" | tee -a $ENVLOG
echo "JOBSUBPARENTJOBID=$JOBSUBPARENTJOBID" | tee -a $ENVLOG
echo "JOBSUBJOBSECTION=$JOBSUBJOBSECTION" | tee -a $ENVLOG
[ $GRID -eq 0 ] && JOBSUBJOBID="noGrid"
echo -e "JOBSUBJOBID=$JOBSUBJOBID\n" | tee -a $ENVLOG

#######################################################
# setup the user local g-2 software if using local area
#######################################################
if [ ${USE_LOCAL_SETUP} -eq 1 ] ; then

   LOCAL_PRODUCTS_DIR=localProducts_gm2_${RELEASE}_prof
   echo -e "\n>>>Setup localProducts:" | tee -a $ENVLOG
   if [ $GRID -eq 0 ]; then
        LOCAL_PRODUCTS_DIR=${MRB_TOP}/${LOCAL_PRODUCTS_DIR}
   else
        LOCAL_PRODUCTS_DIR=${_CONDOR_JOB_IWD}/${INSTALL_AREA}
   	echo -e "The local tar file is: $INPUT_TAR_FILE" | tee -a $ENVLOG
        echo -e "The local product area is: $LOCAL_PRODUCTS_DIR" | tee -a $ENVLOG
   fi
   if [ -d ${LOCAL_PRODUCTS_DIR} ] ; then
   	[ $(/bin/ls -d ${LOCAL_PRODUCTS_DIR}/* | wc -l) -le 1 ] && echo -e "\n!!!Did not find any local packages in ${LOCAL_PRODUCTS_DIR}!!!" | tee -a $ENVLOG && ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs && exit 1
        export PRODUCTS=${LOCAL_PRODUCTS_DIR}:${PRODUCTS}
        echo "PRODUCTS=${PRODUCTS}" | tee -a $ENVLOG
   
        if [[ $RELEASE == v6* || $RELEASE == v7* ]] ; then
         	QUAL="e10:prof"
        fi                

        dir="${LOCAL_PRODUCTS_DIR}/"
        for name in ${LOCAL_PRODUCTS_DIR}/* ; do
     		name="${name#$dir}"

     		[[ $name != "setup" ]] && echo -e "\npackage in local area: ${name}" | tee -a $ENVLOG


                if [ $GRID -eq 0 ]; then
     		     if [[ "${name}" == "gm2geom" ]] ; then
                           echo "Upating ${LOCAL_PRODUCTS_DIR}/$name/${RELEASE}/fcl/geom ..." 
                           for f in `ls ${LOCAL_PRODUCTS_DIR}/$name/${RELEASE}/fcl/geom/*.fcl 2>/dev/null`; do
                               b=$(basename $f)
                               c=$(find ${MRB_TOP}/srcs/$name -name $b 2>/dev/null)
                               if [[ ! -z $c ]]; then
                                     diff -q $f $c > /dev/null 2>&1
                                     [ $? -ne 0 ] && echo "cp -af $c $f" && cp -af $c $f
                               fi
                           done                     
                     else
                           echo "Upating ${LOCAL_PRODUCTS_DIR}/$name/${RELEASE}/fcl ..." 
                           for f in `ls ${LOCAL_PRODUCTS_DIR}/$name/${RELEASE}/fcl/*.fcl 2>/dev/null`; do
                               b=$(basename $f)
                               if [ -e ${MRB_TOP}/srcs/$name/fcl/$b ]; then
                                    diff -q $f ${MRB_TOP}/srcs/$name/fcl/$b > /dev/null 2>&1
                                    [ $? -ne 0 ] && echo "cp -af ${MRB_TOP}/srcs/$name/fcl/$b $f" && cp -af ${MRB_TOP}/srcs/$name/fcl/$b $f
                               fi
                           done
                     fi
                fi

     		if   [[ "${name}" == "artg4" ]] ; then 
         		echo setup artg4 ${RELEASE} -q $QUAL | tee -a $ENVLOG
         		setup artg4 ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2dataproducts" ]] ; then
         		echo setup gm2dataproducts ${RELEASE} -q $QUAL | tee -a $ENVLOG 
        		setup gm2dataproducts ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2midastoart" ]] ; then
                	if [[ "${RELEASE}" == "v7_05_00" ]] ; then
        			echo setup gm2midastoart v4_00_01 -q $QUAL | tee -a $ENVLOG
        			setup gm2midastoart v4_00_01 -q $QUAL
                        else
                        	echo setup gm2midastoart ${RELEASE} -q $QUAL | tee -a $ENVLOG
                        	setup gm2midastoart ${RELEASE} -q $QUAL
                        fi
     		elif [[ "${name}" == "gm2unpackers" ]] ; then
                	if [[ "${RELEASE}" == "v7_05_00" ]] ; then
        			echo setup gm2unpackers v4_00_00 -q $QUAL | tee -a $ENVLOG
        			setup gm2unpackers v4_00_00 -q $QUAL
                        else
                             	echo setup gm2unpackers ${RELEASE} -q $QUAL | tee -a $ENVLOG
                             	setup gm2unpackers ${RELEASE} -q $QUAL
			fi
     		elif [[ "${name}" == "gm2util" ]] ; then
        			echo setup gm2util ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2util ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2calo" ]] ; then
        			echo setup gm2calo ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2calo ${RELEASE} -q $QUAL
	        elif [[ "${name}" == "gm2reconeast" ]] ; then
                    		echo setup gm2reconeast ${RELEASE} -q $QUAL | tee -a $ENVLOG
				setup gm2reconeast ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2tracker" ]] ; then
        			echo setup gm2tracker ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2tracker ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2ringsim" ]] ; then
        			echo setup gm2ringsim ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2ringsim ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2geom" ]] ; then
        			echo setup gm2geom ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2geom ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2aux" ]] ; then
        			echo setup gm2aux ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2aux ${RELEASE} -q $QUAL
     		elif [[ "${name}" == "gm2db" ]] ; then
        			echo setup gm2db ${RELEASE} -q $QUAL | tee -a $ENVLOG
        			setup gm2db ${RELEASE} -q $QUAL
                elif [[ "${name}" == "gm2analyses" ]] ; then
                                echo setup gm2analyses ${RELEASE} -q $QUAL | tee -a $ENVLOG
                                setup gm2analyses ${RELEASE} -q $QUAL
     		fi
   	done

   else
    	echo "The local products area: ${LOCAL_PRODUCTS_DIR} does not exist." | tee -a $ENVLOG
        ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs 
        exit 1
   fi

   ups active >> $ENVLOG

fi

if [[ "${DATA_TYPE}" == "daq" && "${RELEASE}" == "v7_05_00" ]] ; then
   echo setup gm2trackerdaq ${RELEASE} -q e10:prof | tee -a $ENVLOG
   setup gm2trackerdaq ${RELEASE} -q e10:prof
fi

####################################################
# get the environment variables
####################################################
env >> $ENVLOG
echo -e "\n>>>Environment setup saved to env_*.log".  
date -u | tee -a $ENVLOG

####################################################
# copy the environment file
####################################################

function copy_ENVOUTPUT {

echo -e "\n>>>Copying over environment log to ${OUTPUT_DIR}/logs ...\n" | tee -a $ENVLOG
echo -e "ifdh cp -D $ENVLOG ${OUTPUT_DIR}/logs" | tee -a $ENVLOG

if ! ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs; then
   echo -e "\n!!!Failed to copy $ENVLOG to ${OUTPUT_DIR}/logs !!!\n" | tee -a $ENVLOG
else
     [ $CLEAN -eq 1 ] && [ $GRID -eq 1 ] && rm -f $ENVLOG
fi      
}

####################################################
# Function: set ifdh status and clean up
####################################################
function ifdh_endPS()
{
      if [ $USESAMD -eq 1 ]; then

           if [[ $STATUS -eq 0 ]]; then
                echo -e "\n>>>ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} completed" | tee -a $ENVLOG 
                ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} completed 2>&1 | tee -a $ENVLOG
           else
                echo -e "\n>>>ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} bad" | tee -a $ENVLOG 
                ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} bad 2>&1 | tee -a $ENVLOG
           fi

           echo -e "\n>>>ifdh endProcess ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}\n" | tee -a $ENVLOG 
           ifdh endProcess ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} 2>&1 | tee -a $ENVLOG

           date -u | tee -a $ENVLOG
      fi

# copy environment log file with final information
      if [[ -z $1 ]] || [ $1 == "NO" ]; then
           copy_ENVOUTPUT
      fi

      ifdh cleanup

} 

function ifdh_endPJ()
{
      if [ $USESAMD -eq 1 ]; then

           if [[ $STATUS -eq 0 ]]; then
                echo -e "\n>>>ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} completed" | tee -a $ENVLOG 
                ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} completed 2>&1  | tee -a $ENVLOG
           else
                echo -e "\n>>>ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} bad" | tee -a $ENVLOG 
                ifdh setStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} bad 2>&1 | tee -a $ENVLOG
           fi

           echo -e "\n>>>ifdh endProcess ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}" | tee -a $ENVLOG 
           ifdh endProcess ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} 2>&1 | tee -a $ENVLOG

           echo -e "\n>>>ifdh endProject ${SAM_PROJECT_URL}\n" | tee -a $ENVLOG
           ifdh endProject ${SAM_PROJECT_URL} 2>&1 | tee -a $ENVLOG

           date -u | tee -a $ENVLOG
      fi

# copy environment log file with final information
      if [[ -z $1 ]] || [ $1 == "NO" ]; then
           copy_ENVOUTPUT
      fi

      ifdh cleanup

}

####################################################
# Function: get the run and subrun number
####################################################

function get_RUNINFO() {

   if [[ -z $1 ]]; then
      echo -e "\n!!!FILE_URI is empty, cannot get RUNINFO!!!\n" | tee -a $ENVLOG
      STATUS=1
      ifdh_endPS 
      exit 1
   fi

   # save previous run/subrun
   OLDRUN="$RUN.$SUBRUN"
   RUN=""
   SUBRUN=""

   # remove SAM characters from filename  
   arr=(${1//// })
   loc=$(expr ${#arr[@]} - 1)
   NAME=${arr[${loc}]}

   narr=(${NAME//-/ })
   nloc=$(expr ${#narr[@]} - 1)
   NAME=${narr[${nloc}]}

   # get the run and subrun numbers from the filename
   if [[ "${DATA_TYPE}" == "mc" ]] ; then
      marr=(${NAME//_/ })
      mloc=$(expr ${#marr[@]} - 1)
      EXT=${marr[${mloc}]}

      RUN=$(echo ${EXT} | cut -d. -f1)
      SUBRUN=$(echo ${EXT} | cut -d. -f2)

   elif [[ "${DATA_TYPE}" == daq ]] ; then
      if [[ "${NAME}" == gm2_* ]] ; then
         NAME=$(echo ${NAME/gm2_r/r})
      fi

      if [[ "${NAME}" == r* ]] ; then
         RUN=$(echo ${NAME} | cut -d_ -f1 | cut -dn -f2)
         SUBRUN=$(echo ${NAME} | cut -d_ -f2 | cut -d. -f1)
      else
         narr=(${NAME//_/ })
         nloc=$(expr ${#narr[@]} - 1)
         NAME=${narr[${nloc}]}
         narr=(${NAME//./ })
         RUN=${narr[0]}
         SUBRUN=${narr[1]}
      fi
   else 
      echo -e "\n!!!Unknown data type: ${DATA_TYPE}!!!\n" | tee -a $ENVLOG 
      STATUS=1
      ifdh_endPS
      exit 1
   fi


   # check run/subrun number
   if [[ -z $RUN ]] || [[ $RUN =~ [^0-9]+ ]]; then
      echo -e "\n!!!Invalid run number '$RUN' from FILE_URI '$1' !!!\n" | tee -a $ENVLOG
      if [[ $2 == "YES" ]]; then
            return 1
      else
            STATUS=1
            ifdh_endPS
            exit 1
      fi
   fi

   if [[ -z $SUBRUN ]] || [[ $SUBRUN =~ [^0-9]+ ]]; then
      echo -e "\n!!!Invalid subrun number '$SUBRUN' from FILE_URI '$1' !!!\n" | tee -a $ENVLOG
      if [[ $2 == "YES" ]]; then
            return 1
      else
            STATUS=1
            ifdh_endPS
            exit 1
      fi

   fi

   # extract the gun type from the file name
   if [[ "${DATA_TYPE}" == "mc" && "${GUN}" == "None" ]] ; then
      GUN=$(echo $NAME | cut -d_ -f3)
   fi

}


###########################################################
#  Processing input file or input filelist or dataset
###########################################################

if [ $USESAMD -eq 0 ]; then

   SUBRUN=$(($PROCESS + 1))


   # Overwrite the subrun when running in production mode. This is needed to make sure the eventID is unique. art
   # does not like merging events with the same eventID.
   if [ $ROLE == "Production" ]; then
	SUBRUN=$(($JOBSUBJOBSECTION + 1))
   fi


   # Alternative way to get RUN and SUBRUN, replaced by get_RUNINFO, more universal
   # midas files
   #if [[ "${INPUT_FILE}" == *.mid ]]; then
   #        RUN=$(echo ${INPUT_FILE} | sed 's/\(.*\)run\([0-9]\+\).\([0-9]\+\).mid/\2/g')
   #        SUBRUN=$(echo ${INPUT_FILE} | sed 's/\(.*\)run\([0-9]\+\).\([0-9]\+\).mid/\3/g')

   # root files
   #elif [[ "${INPUT_FILE}" == *.root ]]; then
   #        RUN=$(echo ${INPUT_FILE} | sed 's/\(.*\)_\([0-9]\+\).\([0-9]\+\).root/\2/g')
   #        SUBRUN=$(echo ${INPUT_FILE} | sed 's/\(.*\)_\([0-9]\+\).\([0-9]\+\).root/\3/g')

   # get run number if there is an input file
   if [[ "${INPUT_FILE}" != "None" ]]; then           
           get_RUNINFO ${INPUT_FILE}

   # input list
   elif [[ ${INPUT_FILELIST} != "None" ]]; then
           BEGIN=0
           END=0
   
   # split filelist if necessary
           if [[ ! -z $DAGMANJOBID ]]; then

   # DAG job
                BEGIN=$(( 1 + (JOBSUBJOBSECTION-1)*NPERJOB ))
           else
   # non-DAG job
                BEGIN=$(( 1 + PROCESS*NPERJOB ))
           fi

           END=$(( BEGIN + NPERJOB - 1 ))
           if [ $BEGIN -gt 0 -a $END -gt 0 -a $BEGIN -le $END ]; then
                sed -n "$BEGIN,$END p;$END q" ${INPUT_FILELIST} > ${INPUT_FILELIST}.$BEGIN.$END
                INPUT_FILELIST=${INPUT_FILELIST}.$BEGIN.$END
           fi

           FIRST=$(cat ${INPUT_FILELIST} | head -1)
           if [[ ! -z $FIRST ]]; then
                 get_RUNINFO $FIRST
           else
                 echo -e "\n!!!Did not find a file for ${INPUT_FILELIST}, check logs to see if files have been consumed already!!!" | tee -a $ENVLOG
                 ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
                 exit 0
           fi                   
   fi


else

   # start a project (when running on the grid -- jobsub does this for you)
   if [ $GRID -eq 0 ]; then
      echo -e "\n>>>ifdh startProject ${PROJECTNAME} gm2 ${SAMDATASET} ${USER} gm2\n" | tee -a $ENVLOG 
      if ! SAM_PROJECT_URL=$(ifdh startProject ${PROJECTNAME} gm2 ${SAMDATASET} ${USER} gm2 2>&1); then 
         echo -e "\n!!!Unable to start a SAM project!!!\nError:" | tee -a $ENVLOG
         echo ${SAM_PROJECT_URL} | tee -a $ENVLOG
         ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
         exit 1
      fi

   # find a project (the job is running on the grid -- therefore jobsub owns the project)
   else
      echo -e "\n>>>ifdh findProject ${SAM_PROJECT_NAME} gm2\n" | tee -a $ENVLOG
      if ! SAM_PROJECT_URL=$(ifdh findProject ${SAM_PROJECT_NAME} gm2 2>&1); then
         echo -e "\n!!!Unable to find a SAM project!!!\nError:" | tee -a $ENVLOG
         echo ${SAM_PROJECT_URL} | tee -a $ENVLOG
         ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
         exit 1
      fi

   fi

   # create a process id for the SAM project
   echo -e "\n>>>ifdh establishProcess ${SAM_PROJECT_URL} gm2 $RELEASE $HOST $USER $CAMPAIGN $JOBSUBJOBID ${SAM_FILE_LIMIT} $SCHEMAS\n"  | tee -a $ENVLOG
   if ! SAM_PROCESS_ID=$(ifdh establishProcess ${SAM_PROJECT_URL} gm2 $RELEASE $HOST $USER $CAMPAIGN $JOBSUBJOBID ${SAM_FILE_LIMIT} $SCHEMAS 2>&1); then
   # try again in case of 404 or 503
      sleep 1
      if ! SAM_PROCESS_ID=$(ifdh establishProcess ${SAM_PROJECT_URL} gm2 $RELEASE $HOST $USER $CAMPAIGN $JOBSUBJOBID ${SAM_FILE_LIMIT} $SCHEMAS 2>&1); then
         echo -e "\n!!!Unable to establish a SAM process!!!\nError:" | tee -a $ENVLOG
         echo ${SAM_PROCESS_ID} | tee -a $ENVLOG
         ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
         exit 1
      fi
   fi

   # grab the next file in line to extract useful metadata
   echo -e "\n>>>ifdh getNextFile ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}\n" | tee -a $ENVLOG
   if ! FILE_URI=$(ifdh getNextFile ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} 2>&1); then
      echo -e "\n!!!Did not find a file for the project: ${SAM_PROJECT_URL}, check logs to see if files have been consumed already!!!" | tee -a $ENVLOG
      echo "SAM_PROJECT_URL : ${SAM_PROJECT_URL}" | tee -a $ENVLOG
      echo "SAM_PROCESS_ID  : ${SAM_PROCESS_ID}" | tee -a $ENVLOG
      echo -e "${FILE_URI}\n" | tee -a $ENVLOG
      ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
      ifdh endProcess ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} 
      exit 0
   fi

   echo -e "\n>>>Initial FILE URI to parse: $FILE_URI\n" | tee -a $ENVLOG

   get_RUNINFO ${FILE_URI} 


fi #end of $USESAMD

####################################################
# Define INPUT
####################################################
INPUT=""
if [ $USESAMD -eq 0 ]; then
   if [[ ${INPUT_FILE} != "None" ]]; then
         INPUT=$(basename ${INPUT_FILE})
   else
         INPUT=$(basename ${INPUT_FILELIST})
   fi
else
   INPUT=$SAMDATASET
fi

####################################################
# Function: print out some run info
####################################################

function print_RUNINFO {

echo
echo "INPUT:            ${INPUT}"          | tee -a $ENVLOG
echo "FCL:              ${RFCLNAME}"       | tee -a $ENVLOG     
echo "ROLE:             ${ROLE}"           | tee -a $ENVLOG     
echo "CAMPAIGN:         ${CAMPAIGN}"       | tee -a $ENVLOG
echo "RUN:              ${RUN}"            | tee -a $ENVLOG
echo "SUBRUN:           ${SUBRUN}"         | tee -a $ENVLOG
echo "TYPE:             ${DATA_TYPE}"      | tee -a $ENVLOG
echo "TIER:             ${TIER}"           | tee -a $ENVLOG
echo "GUN:              ${GUN}"            | tee -a $ENVLOG
echo "PARTICLE:         ${PARTICLE}"       | tee -a $ENVLOG
echo "TAG:              ${TAG}"            | tee -a $ENVLOG
echo "SYSTEM:           ${SYSTEM}"         | tee -a $ENVLOG
echo "CLUSTER:          ${CLUSTER}"        | tee -a $ENVLOG
echo "AFFINITY:         ${CORE}"           | tee -a $ENVLOG
echo "SAM_FILE_LIMIT:   ${SAM_FILE_LIMIT}" | tee -a $ENVLOG
echo "IFDH_ART:         ${IFDH_ART}"       | tee -a $ENVLOG
echo "ONE_BY_ONE:       ${ONE_BY_ONE}"     | tee -a $ENVLOG
[[ ${PROCESS_NAME} != "None" ]] && echo "PROCESS NAME:     ${PROCESS_NAME}" | tee -a $ENVLOG     
[[ $ROLE == "Production" ]] &&     echo "REQUEST ID:       ${REQUEST}"      | tee -a $ENVLOG     
[[ ! -z $SCHEMAS ]]         &&     echo "SCHEMAS:          ${SCHEMAS}"      | tee -a $ENVLOG
[ ${MEMORY_TRACKER} -eq 1 ] &&     echo "MEMORY TRACKER:   YES"             | tee -a $ENVLOG
[ $KICKER -eq 0 ]           &&     echo "KICKER:           OFF"             | tee -a $ENVLOG
[ $QUAD   -eq 0 ]           &&     echo "QUAD:             OFF"             | tee -a $ENVLOG
[ ${JSON} -eq 1 ]           &&     echo "JSON:             YES"             | tee -a $ENVLOG
echo "OUTPUT_DIR:       ${OUTPUT_DIR}"     | tee -a $ENVLOG
echo

}


# rename ENVLOG
mv $ENVLOG env_${CLUSTER}_run${RUN}.${SUBRUN}.log
ENVLOG="env_${CLUSTER}_run${RUN}.${SUBRUN}.log"

print_RUNINFO

for var in RUN SUBRUN DATA_TYPE TIER TAG PARTICLE CLUSTER; do
	if [[ -z ${!var} ]] || [[ ${!var} == "None" ]]; then
        	echo -e "\n!!!$var is NOT defined!!!\n" | tee -a $ENVLOG
                ifdh_endPJ
		exit 1
	fi
done                

if [[ ${DATA_TYPE} == "mc" ]] && ( [[ -z $GUN ]] || [[ $GUN == "None" ]] ); then
        	echo -e "\n!!!GUN is NOT defined!!!\n" | tee -a $ENVLOG
                ifdh_endPJ
		exit 1
fi

if [[ $CORE != "None" ]]; then
   TASKSET="taskset -c $CORE"
   $TASKSET echo 
   if [ $? -ne 0 ]; then
        echo -e "\n!!!Unable to run 'taskset -c $CORE echo'!!!\n" | tee -a $ENVLOG
        ifdh_endPJ
	exit 1
   fi
fi


###########################################################
#  Processing fhicl file
###########################################################

echo -e "\n>>>Processing ${RMAINFCLNAME} ...\n" | tee -a $ENVLOG

# check if valid TAG
tagstring1="gm2${TAG}_particle_gun.root"
tagstring2="gm2${TAG}_particle_gun_${TIER}.root"
tagstring3="gm2${TAG}_${TIER}.root"


if [[ "${DATA_TYPE}" == "mc" ]] ; then

	grep -i -q "$tagstring1\|$tagstring2" ${RMAINFCLNAME}
	if [ $? -ne 0 ]; then
     		echo -e "\n!!!Unsupported TAG \"$TAG\" for data type \"${DATA_TYPE}\", tier \"$TIER\", INCLUDE_ANA=${INCLUDE_ANA}, check ${RMAINFCLNAME} !!!" | tee -a $ENVLOG
     		echo -e "The ${DATA_TYPE} ${TIER} fhicl file must contain one of the following syntax:" | tee -a $ENVLOG
     		echo -e "\t1. $tagstring1" | tee -a $ENVLOG
     		echo -e "\t2. $tagstring2" | tee -a $ENVLOG
     		ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
                STATUS=1
                ifdh_endPJ 
		exit 1
	fi

	grep -i -q "${tagstring1/.root/.log}\|${tagstring2/.root/.log}" ${RMAINFCLNAME}
	if [ $? -ne 0 ]; then
     		echo -e "\n!!!The ${DATA_TYPE} ${TIER} fhicl file {RMAINFCLNAME} must contain one of the following syntax:" | tee -a $ENVLOG
     		echo -e "\t1. ${tagstring1/.root/.log}" | tee -a $ENVLOG
     		echo -e "\t2. ${tagstring2/.root/.log}" | tee -a $ENVLOG
     		ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
                STATUS=1
                ifdh_endPJ 
     		exit 1
	fi  

else 

	grep -i -q "$tagstring3" ${RMAINFCLNAME}
	if [ $? -ne 0 ]; then
     		echo -e "\n!!!Unsupported TAG \"$TAG\" for data type \"${DATA_TYPE}\", tier \"$TIER\", INCLUDE_ANA=${INCLUDE_ANA}, check ${RMAINFCLNAME} !!!" | tee -a $ENVLOG
     		echo -e "The ${DATA_TYPE} ${TIER} fhicl file must contain the following syntax:" | tee -a $ENVLOG
     		echo -e "\t$tagstring3\n" | tee -a $ENVLOG
     		ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
                STATUS=1
                ifdh_endPJ 
     		exit 1
	fi

	grep -i -q "${tagstring3/.root/.log}" ${RMAINFCLNAME}
	if [ $? -ne 0 ]; then
     		echo -e "\n!!!The ${DATA_TYPE} ${TIER} fhicl file ${RMAINFCLNAME} must contain the following syntax:" | tee -a $ENVLOG
     		echo -e "\t${tagstring3/.root/.log}\n" | tee -a $ENVLOG
     		ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
                STATUS=1
                ifdh_endPJ 
     		exit 1
	fi


fi


# search, replace, overwrite contents of the fhicl file
if [ $USESAMD -eq 1 ] ; then
   if [[ -z $SAM_FILE_LIMIT ]] || [[ $SAM_FILE_LIMIT -le 0 ]]; then
      sed -e "s,^.*maxInputFiles.*$,,g" -i ${RMAINFCLNAME}
   else
      sed -e "s,maxInputFiles.*$,maxInputFiles : ${SAM_FILE_LIMIT},g" -i ${RMAINFCLNAME}
   fi
fi

# update process name if asked
if [[ ${PROCESS_NAME} != "None" ]]; then
      sed -e "s,process_name.*$,process_name : ${PROCESS_NAME},g" -i ${FCLNAME}
fi


if [[ ${DATA_TYPE} == "mc" ]] ; then
# MC
      if [[ ${TIER} != "ana" ]]; then
# Production mode
# Modify the first FCL
         sed -e "s,source.firstRun.*$,source.firstRun : ${RUN},g" \
             -e "s,source.firstSubRun.*$,source.firstSubRun : ${SUBRUN},g" \
             -e "s,requestid.*$,requestid : \"${REQUEST}\",g" \
             -e "s,run_config.*$,run_config : \"${CONFIG}\",g" \
             -e "s,campaign.*$,campaign : \"${CAMPAIGN}\" } ], g" \
             -e "s,applicationVersion.*$,applicationVersion : \"${RELEASE}\",g" \
             -e "s,StoredMuons.*$,StoredMuons : ${MUONSPERFILL},g" \
             -i ${RMAINFCLNAME}

# Also modify the last FCL
         if [[ "${FCLNAME}" != "${RMAINFCLNAME}" ]]; then
            sed -e "s,source.firstRun.*$,source.firstRun : ${RUN},g" \
                -e "s,source.firstSubRun.*$,source.firstSubRun : ${SUBRUN},g" \
                -e "s,requestid.*$,requestid : \"${REQUEST}\",g" \
                -e "s,run_config.*$,run_config : \"${CONFIG}\",g" \
                -e "s,campaign.*$,campaign : \"${CAMPAIGN}\" } ], g" \
                -e "s,applicationVersion.*$,applicationVersion : \"${RELEASE}\",g" \
                -e "s,StoredMuons.*$,StoredMuons : ${MUONSPERFILL},g" \
                -i ${FCLNAME}
         fi
 
         [[ ${INCLUDE_ANA} -eq 1 ]] && TIER="ana"         

         sed -e "s,gm2${TAG}_particle_gun.log,gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_particle_gun.root,gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_particle_gun.fcl,gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).log,gm2${TAG}_${PARTICLE}_${GUN}_\1_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).root,gm2${TAG}_${PARTICLE}_${GUN}_\1_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).fcl,gm2${TAG}_${PARTICLE}_${GUN}_\1_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -i ${RMAINFCLNAME}

      	 OUTPUT_ROOT=gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.root
         OUTPUT_LOG=gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.log
         OUTPUT_FCL=gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.fcl
         OUTPUT_DB=gm2${TAG}_${PARTICLE}_${GUN}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.db

      else
# Analyzer mode 
         sed -e "s,gm2${TAG}_particle_gun.log,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_particle_gun.root,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_particle_gun.fcl,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).log,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).root,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_particle_gun_\([^\]*\).fcl,gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -e "s,applicationVersion.*$,applicationVersion : \"${RELEASE}\",g" \
             -e "s,StoredMuons.*$,StoredMuons : ${MUONSPERFILL},g" \
             -i ${RMAINFCLNAME}

         OUTPUT_ROOT=gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.root
         OUTPUT_LOG=gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.log
         OUTPUT_FCL=gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.fcl
         OUTPUT_DB=gm2${TAG}_${PARTICLE}_${GUN}_ana_${CLUSTER}_${RUN}.${SUBRUN}.db

      fi


else
# DATA
      if [[ "${TIER}" != "ana" ]] ; then
# Production mode
         sed -e "s,requestid.*,requestid : \"${REQUEST}\",g" \
             -e "s,run_config.*,run_config : \"${CONFIG}\",g" \
             -e "s,campaign.*$,campaign : \"${CAMPAIGN}\" } ], g" \
             -e "s,applicationVersion.*,applicationVersion : \"${RELEASE}\",g" \
             -i ${RMAINFCLNAME}

         if [[ "${FCLNAME}" != "${RMAINFCLNAME}" ]]; then
            sed -e "s,requestid.*,requestid : \"${REQUEST}\",g" \
                -e "s,run_config.*,run_config : \"${CONFIG}\",g" \
                -e "s,campaign.*$,campaign : \"${CAMPAIGN}\" } ], g" \
                -e "s,applicationVersion.*$,applicationVersion : \"${RELEASE}\",g" \
                -i ${FCLNAME}
         fi

         [[ ${INCLUDE_ANA} -eq 1 ]] && TIER="ana"         

         sed -e "s,gm2${TAG}_\([^_]*\).log,gm2${TAG}_\1_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_\([^_]*\).root,gm2${TAG}_\1_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_\([^_]*\).fcl,gm2${TAG}_\1_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -i ${RMAINFCLNAME}

         OUTPUT_ROOT=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.root
         OUTPUT_LOG=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.log
         OUTPUT_FCL=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.fcl
         OUTPUT_DB=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.db
         OUTPUT_JSON=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.json

      else
# Analyzer mode
      	 sed -e "s,gm2${TAG}_ana.log,gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.log,g" \
             -e "s,gm2${TAG}_ana.root,gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.root,g" \
             -e "s,gm2${TAG}_ana.fcl,gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.fcl,g" \
             -i ${RMAINFCLNAME}
 
         OUTPUT_ROOT=gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.root
         OUTPUT_LOG=gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.log
         OUTPUT_FCL=gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.fcl
         OUTPUT_DB=gm2${TAG}_ana_${CLUSTER}_${RUN}.${SUBRUN}.db
         OUTPUT_JSON=gm2${TAG}_${TIER}_${CLUSTER}_${RUN}.${SUBRUN}.json

      fi

      # old data

      if [[ "${TIER}" == "unpack" || "${TIER}" == "full" ]] ; then

            run=`expr ${RUN} + 0`
            max=902
            if [[ ${run} -gt 0 ]] && [[  "${run}" -le "${max}" ]] ; then 

                 sed -e "s,sequence\:\:common,sequence\:\:commonNoCompressed,g" \
                     -e "s,inputModuleLabel.*,inputModuleLabel : MidasBankInput,g" \
                     -i ${RMAINFCLNAME}

                 echo "" >> ${RMAINFCLNAME}
                 echo "source.detail.requestedBanks : [ \"STRW\"," >> ${RMAINFCLNAME}
                 echo "		\"TRIG\", \"GPS0\"," >> ${RMAINFCLNAME}
                 echo "		\"TTCA\", \"TTCB\", \"TTCZ\"," >> ${RMAINFCLNAME}
                 echo "		\"CC\", \"LC\", \"PC\", \"AC\"," >> ${RMAINFCLNAME}
                 echo "         \"CA\", \"LA\", \"PA\", \"AA\"," >> ${RMAINFCLNAME}
                 echo "         \"CZ\", \"LZ\", \"PZ\", \"AZ\"," >> ${RMAINFCLNAME}
                 echo "		\"CQ\", \"HQ\", \"0Q\"," >> ${RMAINFCLNAME}
                 echo "		\"CB\", \"LB\", \"PB\", \"AB\"," >> ${RMAINFCLNAME}
                 echo "		\"CR\", \"LR\", \"PR\", \"AR\"," >> ${RMAINFCLNAME}
                 echo "		\"CF\", \"CP\", \"CS\", \"CT\" ]" >> ${RMAINFCLNAME} 

            else 
                 echo -e "RUN $run has compressed data objects\n" | tee -a $ENVLOG
            fi

            if [[ "${RELEASE}" == "v7_06_02" || "${RELEASE}" == "v7_06_01" ]] ; then
                  cqfilter=1589
                  if [ "${run}" -ge "${cqfilter}" ] ; then
                     sed -e "s,\[muonPath,\[muonPathNoCQ,g" \ 
                         -i ${RMAINFCLNAME} 
                  fi
            fi # end of if RELEASE

      fi # end of "old data"

fi # end of if DATA_TYPE

# remove message_categories for debugging
if [ $VERBOSE -eq 1 ]; then
     grep -q "${message_categories}" ${RMAINFCLNAME}
     if [ $? -ne 0 ]; then
          echo -e "\n***${RMAINFCLNAME} does not seem to contain ${message_categories}, verbose printout already?***\n" | tee -a $ENVLOG
     else
          sed "s/.\+${message_categories}.*//g" -i ${RMAINFCLNAME}
     fi
fi     

# add memory tracker if not already
if [ ${MEMORY_TRACKER} -eq 1 ]; then
     grep -q "MemoryTracker" ${RMAINFCLNAME}
     if [ $? -ne 0 ]; then
         # this assumes we have "services: {" pattern in the fhiCl file
         sed '/^services\s*:\s*{/a\\n\
         MemoryTracker : {\
           dbOutput: {\
             filename: "'${OUTPUT_DB}'"\
             overwrite: false\
           }\
         includeMallocInfo: true\
         }' -i ${RMAINFCLNAME}
         grep -q "MemoryTracker" ${RMAINFCLNAME}
         if [ $? -ne 0 ]; then
 	      echo -e "\n!!!Unable to add MemoryTracker service to ${RMAINFCLNAME}, check if it contains \"services: {\" pattern!!!\n" | tee -a $ENVLOG
     	      ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
      	      ifdh_endPJ
              exit 1 
         fi  
     else
         sed 's/includeMallocInfo: false/includeMallocInfo: true/g' -i ${RMAINFCLNAME}
         sed '/MemoryTracker/,+5s/filename:\(.*\).db\(.*\)/filename: "'${OUTPUT_DB}'"/g' -i ${RMAINFCLNAME}
     fi
fi


# update json information
if [ ${JSON} -eq 1 ] ; then
   echo " " >> ${RMAINFCLNAME}
   echo "services.ConDBJsonFileWriter.runOnGrid : true" >> ${RMAINFCLNAME}
   echo "services.ConDBJsonFileWriter.writeScratchArea : false" >> ${RMAINFCLNAME}
   echo "services.ConDBJsonFileWriter.outFilename : \"${OUTPUT_JSON}\" " >> ${RMAINFCLNAME}
fi

# add text to file
grep -i -q "source.firstRun" ${RMAINFCLNAME}
if [[ $? -ne 0 && "${TIER}" != "ana" && ${SAM_FILE_LIMIT} -eq 1 ]]; then
      echo "" >> ${RMAINFCLNAME}
      echo "source.firstRun : ${RUN}" >> ${RMAINFCLNAME} 
      echo "source.firstSubRun : ${SUBRUN}" >> ${RMAINFCLNAME}
fi


# add text to file
if [[ ${SYSTEM} == "harp" ]] ; then
      echo "${services}.Geometry.fiberHarp.rOffset : 0" >> ${RMAINFCLNAME}
      sed -e "s,ghostFiberHarp.rOffset.*,ghostFiberHaper.rOffset : 0,g" \
          -e "s,${services}.Geometry.ghostFiberHarp.ghostFiberHarp_geom.rOffset.*,${services}.Geometry.ghostFiberHarp.ghostFiberHarp_geom.rOffset : 0,g" \
          -i ${RMAINFCLNAME}
      if [[ ${CONFIG} == "harpCal" ]] ; then
         echo "" >> ${RMAINFCLNAME}
         echo "${services}.Geometry.fiberHarp.harpType : [ \"HARP_Y_CAL\", \"HARP_X_CAL\", \"HARP_Y_CAL\", \"HARP_X_CAL\" ]" >> ${RMAINFCLNAME} 
         echo "${services}.Geomerty.ghostFiberHarp.harpType : [ \"HARP_Y_CAL\", \"HARP_X_CAL\", \"HARP_Y_CAL\", \"HARP_X_CAL\" ]" >> ${RMAINFCLNAME}
      fi
fi  

# remove kicker if requested
if [ $KICKER -eq 0 ]; then
     grep -q "$kickerpath" ${RMAINFCLNAME}
     if [ $? -ne 0 ]; then
          echo -e "\n***${RMAINFCLNAME} does not seem to contain '$kickerpath', Kicker OFF already?***\n" | tee -a $ENVLOG
     else
          sed -e "s/\(.*\)\s\+.*${kickerpath}.*,/\1/g"  -e "s/\[\(.*\)\s\+.*${kickerpath}[^,]\+]/[\1]/g" -e "s/,]/]/g" -i ${RMAINFCLNAME}
     fi
     grep -q "$kickerpath" ${RMAINFCLNAME}
     if [ $? -eq 0 ]; then
 	      echo -e "\n!!!Unable to find and remove '$kickerpath' from ${RMAINFCLNAME}!!!\n" | tee -a $ENVLOG
     	      ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
      	      ifdh_endPJ
              exit 1 
     fi
fi     

# remove quad if requested
if [ $QUAD -eq 0 ]; then
     grep -q "$quadpath" ${RMAINFCLNAME}
     if [ $? -ne 0 ]; then
          echo -e "\n***${RMAINFCLNAME} does not seem to contain '$quadpath', Quad OFF already?***\n" | tee -a $ENVLOG
     else
          sed -e "s/\(.*\)\s\+.*${quadpath}.*,/\1/g"  -e "s/\[\(.*\)\s\+.*${quadpath}[^,]\+]/[\1]/g" -e "s/,]/]/g" -i ${RMAINFCLNAME}
     fi
     grep -q "$quadpath" ${RMAINFCLNAME}
     if [ $? -eq 0 ]; then
 	      echo -e "\n!!!Unable to find and remove '$quadpath' from ${RMAINFCLNAME}!!!\n" | tee -a $ENVLOG
     	      ifdh cp -D $RMAINFCLNAME  ${OUTPUT_DIR}/fcl
      	      ifdh_endPJ
              exit 1 
     fi
fi     


if [[ ${PARTICLE} != "muon" ]] ; then
      echo "${services}.TheBeam.particles : [ \"${PARTICLE}\" ]" >> ${RMAINFCLNAME}
fi

if [[ ${INCLUDE_ANA} -eq 1 ]] ; then
      sed -e "s,\/\/#,"",g" \
          -i ${RMAINFCLNAME}
fi

# rename the fhicl file
echo -e ">>>mv ${RMAINFCLNAME} ${OUTPUT_FCL}\n" | tee -a $ENVLOG
mv ${RMAINFCLNAME} ${OUTPUT_FCL}


####################################################
# Function: update run and subrun number if needed
####################################################

function update_RUNOUTPUT {

  OUTPUT_ROOT=$(echo ${OUTPUT_ROOT} | sed "s/[0-9]\+.[0-9]\+.root/${RUN}.${SUBRUN}.root/g")
  OUTPUT_FCL=${OUTPUT_ROOT//.root/.fcl}
  OUTPUT_LOG=${OUTPUT_ROOT//.root/.log}
  OUTPUT_DB=${OUTPUT_ROOT//.root/.db}
  OUTPUT_JSON=${OUTPUT_ROOT//.root/.json}

  if [ ${MULTIPLE_ROOT} == "YES" ]; then
       rm -f *.root
  else
       rm -f ${OUTPUT_ROOT}
  fi
  rm -f ${OUTPUT_LOG}
  rm -f ${OUTPUT_DB}
  
  if [ ${JSON} -eq 1 ] ; then
	rm -f ${OUTPUT_JSON}
  fi 

}


######################################################
# Function: copy output for this run and subrun number
######################################################

function copy_RUNOUTPUT {

echo -e "\n>>>Copying over results to ${OUTPUT_DIR} ...\n" | tee -a $ENVLOG

# copy fcl to dCache
echo ifdh cp -D ${OUTPUT_FCL} ${OUTPUT_DIR}/fcl | tee -a $ENVLOG
ifdh cp -D ${OUTPUT_FCL} ${OUTPUT_DIR}/fcl

# copy log to dCache

if [ -f ${OUTPUT_LOG} ]; then
     echo ifdh cp -D ${OUTPUT_LOG} ${OUTPUT_DIR}/logs | tee -a $ENVLOG
     ifdh cp -D ${OUTPUT_LOG} ${OUTPUT_DIR}/logs | tee -a $ENVLOG
     if [ ${MULTIPLE_FCL} == "YES" ]; then
          OUTPUT_TGZ=${OUTPUT_ROOT//.root/.tgz}
          if [ -f ${OUTPUT_TGZ} ]; then
               echo ifdh cp -D ${OUTPUT_TGZ} ${OUTPUT_DIR}/logs | tee -a $ENVLOG
               ifdh cp -D ${OUTPUT_TGZ} ${OUTPUT_DIR}/logs | tee -a $ENVLOG
          else
               echo -e "\n!!!Output tgz file ${OUTPUT_TGZ} does NOT exist!!!\n" | tee -a $ENVLOG
               STATUS=1
# not in a loop, fail everything and exit
               if [[ ${ONE_BY_ONE} == "NO" ]]; then
                     ifdh_endPS
                     exit 1
               fi
          fi

     fi
     [ $CLEAN -eq 1 ] && rm -f ${OUTPUT_LOG} ${OUTPUT_TGZ}      
else
     echo -e "\n!!!Output log file ${OUTPUT_LOG} does NOT exist!!!\n" | tee -a $ENVLOG
     STATUS=1
# not in a loop, fail everything and exit
     if [[ ${ONE_BY_ONE} == "NO" ]]; then
           ifdh_endPS
           exit 1
     fi
fi 

# copy memory db to dCache

if [ ${MEMORY_TRACKER} -eq 1 ]; then
     if [ -f ${OUTPUT_DB} ]; then
          echo ifdh cp -D ${OUTPUT_DB}  ${OUTPUT_DIR}/logs | tee -a $ENVLOG
          ifdh cp -D ${OUTPUT_DB}  ${OUTPUT_DIR}/logs
          [ $CLEAN -eq 1 ] && rm -f ${OUTPUT_DB}      
     else
          echo -e "\n!!!Output DB file ${OUTPUT_DB} does NOT exist!!!\n" | tee -a $ENVLOG
          STATUS=1
	  date -u | tee -a $ENVLOG
          if [[ ${ONE_BY_ONE} == "NO" ]]; then
                ifdh_endPS
                exit 1
          fi

     fi 
fi


# copy json to dCache

if [ ${JSON} -eq 1 ]; then
        echo -e "\nThe items are : [ $(ls) ].\n" | tee -a $ENVLOG
	if [ -f ${OUTPUT_JSON} ]; then
		echo ifdh cp -D ${OUTPUT_JSON}  ${OUTPUT_DIR}/json | tee -a $ENVLOG
		ifdh cp -D ${OUTPUT_JSON}  ${OUTPUT_DIR}/json
		[ $CLEAN -eq 1 ] && rm -rf ${OUTPUT_JSON}
	else
		echo -e "\n!!!Output JSON file ${OUTPUT_JSON} does NOT exist!!!\n" | tee -a $ENVLOG
		STATUS=1
          	if [[ ${ONE_BY_ONE} == "NO" ]]; then
                	ifdh_endPS
                	exit 1
          	fi
	fi
fi		


# copy root to dCache     

if [ $STATUS != 0 ]; then
     echo -e "\n!!!Command exit status $STATUS != 0, no output root file(s) to copy !!!\n" | tee -a $ENVLOG
     date -u | tee -a $ENVLOG  
     if [[ ${ONE_BY_ONE} == "NO" ]]; then
           ifdh_endPS
           exit 1
     fi

# multiple root output?
elif [ ${MULTIPLE_ROOT} == "YES" ]; then
       OUTPUT_ROOTLIST=$(/bin/ls *.root 2>/dev/null)
       if [[ ! -z ${OUTPUT_ROOTLIST} ]]; then
             echo ifdh cp -D ${OUTPUT_ROOTLIST} ${OUTPUT_DIR}/data | tee -a $ENVLOG
             if ! ifdh cp -D ${OUTPUT_ROOTLIST} ${OUTPUT_DIR}/data | tee -a $ENVLOG; then
                echo -e "\n!!!Failed to copy output root files to ${OUTPUT_DIR}/data !!!\n" | tee -a $ENVLOG
                date -u | tee -a $ENVLOG
                STATUS=1
                if [[ ${ONE_BY_ONE} == "NO" ]]; then
                      ifdh_endPS
                      exit 1
                fi
# must clean up for multiple root output
             else
                rm -f ${OUTPUT_ROOTLIST}
             fi
       else
             echo -e "\n!!!Output root files do NOT exist!!!\n" | tee -a $ENVLOG
             echo -e "Listing current directory content:\n" | tee -a $ENVLOG
             /bin/ls -lart | tee -a $ENVLOG
             date -u | tee -a $ENVLOG
             STATUS=1
             if [[ ${ONE_BY_ONE} == "NO" ]]; then
                ifdh_endPS
                exit 1
             fi
       fi

# multiple fcls?
elif [ ${MULTIPLE_FCL} == "YES" ]; then
       if [ -f ${FINAL_OUTPUT_ROOT} ]; then
            echo ifdh cp -D ${FINAL_OUTPUT_ROOT} ${OUTPUT_DIR}/data | tee -a $ENVLOG
            if ! ifdh cp -D ${FINAL_OUTPUT_ROOT} ${OUTPUT_DIR}/data | tee -a $ENVLOG; then
               echo -e "\n!!!Failed to copy final output file ${FINAL_OUTPUT_ROOT} to ${OUTPUT_DIR}/data !!!\n" | tee -a $ENVLOG
               date -u | tee -a $ENVLOG
               STATUS=1
               if [[ ${ONE_BY_ONE} == "NO" ]]; then
                     ifdh_endPS
                     exit 1
               fi
             elif [ $CLEAN -eq 1 ]; then
               rm -f ${FINAL_OUTPUT_ROOT}
             fi
       else
             echo -e "\n!!!Final output file ${FINAL_OUTPUT_ROOT} does NOT exist!!!\n" | tee -a $ENVLOG
             date -u | tee -a $ENVLOG  
             STATUS=1
             if [[ ${ONE_BY_ONE} == "NO" ]]; then
                   ifdh_endPS
                   exit 1
             fi
       fi
elif [ -f ${OUTPUT_ROOT} ]; then
          echo ifdh cp -D ${OUTPUT_ROOT} ${OUTPUT_DIR}/data | tee -a $ENVLOG
          if ! ifdh cp -D ${OUTPUT_ROOT} ${OUTPUT_DIR}/data | tee -a $ENVLOG; then
               echo -e "\n!!!Failed to copy output file ${OUTPUT_ROOT} to ${OUTPUT_DIR}/data !!!\n" | tee -a $ENVLOG
               date -u | tee -a $ENVLOG
               STATUS=1
               if [[ ${ONE_BY_ONE} == "NO" ]]; then
                     ifdh_endPS
                     exit 1
               fi
          elif [ $CLEAN -eq 1 ]; then
               rm -f ${OUTPUT_ROOT}
          fi

else
          echo -e "\n!!!Output file ${OUTPUT_ROOT} does NOT exist !!!\n" | tee -a $ENVLOG
          date -u | tee -a $ENVLOG  
          STATUS=1
          if [[ ${ONE_BY_ONE} == "NO" ]]; then
                ifdh_endPS
                exit 1
          fi
fi 


echo -e "\n>>>Finished copying over the job log, fcl, and root files to ${OUTPUT_DIR} ...\n" | tee -a $ENVLOG

}

####################################################
# Function: run multiple fcls if required
####################################################
function RUN_FCLS {

if [[ $1 == "YES" ]]; then

     echo -e ">>>ln -sf ${OUTPUT_ROOT} ${RFCLNAMELIST[0]//.fcl/.root}\n" | tee -a $ENVLOG
     ln -sf ${OUTPUT_ROOT} ${RFCLNAMELIST[0]//.fcl/.root} 2>&1 | tee -a $ENVLOG

     for i in `seq 1 $((${#RFCLNAMELIST[@]}-1))`; do
           date -u | tee -a $ENVLOG

# for analyzer mode, last step should use -T
           if [[ $TIER == "ana" ]] && [[ $i -eq $((${#RFCLNAMELIST[@]}-1)) ]]; then
                echo -e ">>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${RFCLNAMELIST[i]} -s ${RFCLNAMELIST[i-1]//.fcl/.root} -T ${RFCLNAMELIST[i]//.fcl/.root} \n" | tee -a $ENVLOG
                $TASKSET gm2 $TRACE $TIMING -c "${RFCLNAMELIST[i]}" -s "${RFCLNAMELIST[i-1]//.fcl/.root}" -T "${RFCLNAMELIST[i]//.fcl/.root}" 2>&1 | tee -a $ENVLOG
           else 
                echo -e ">>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${RFCLNAMELIST[i]} -s ${RFCLNAMELIST[i-1]//.fcl/.root} -o ${RFCLNAMELIST[i]//.fcl/.root} \n" | tee -a $ENVLOG
                $TASKSET gm2 $TRACE $TIMING -c "${RFCLNAMELIST[i]}" -s "${RFCLNAMELIST[i-1]//.fcl/.root}" -o "${RFCLNAMELIST[i]//.fcl/.root}" 2>&1 | tee -a $ENVLOG
           fi

           STATUS=${PIPESTATUS[0]}

           if [[ $STATUS -eq 0 ]]; then
           #  job success
              echo -e ">>>${RFCLNAMELIST[i]} command exit status: ${PIPESTATUS[0]}, success!\n" | tee -a $ENVLOG
              [ $CLEAN -eq 1 ] && rm -f "${RFCLNAMELIST[i-1]//.fcl/.root}"
           else
           #  job fail
              echo -e ">>>${RFCLNAMELIST[i]} command exit status: ${PIPESTATUS[0]}, fail!\n" | tee -a $ENVLOG
              break
           fi
     done


     if [ $STATUS -eq 0 ]; then 
          FINAL_OUTPUT_ROOT=${OUTPUT_ROOT/_ana_/_final-ana_}
          FINAL_OUTPUT_ROOT=${OUTPUT_ROOT/_${TIER}_/_final_}
          echo -e ">>>mv ${RFCLNAMELIST[i]//.fcl/.root} ${FINAL_OUTPUT_ROOT}\n" | tee -a $ENVLOG
          mv ${RFCLNAMELIST[i]//.fcl/.root} ${FINAL_OUTPUT_ROOT} 2>&1 | tee -a $ENVLOG
     fi
     
     if [ $CLEAN -eq 1 ]; then
          rm -f "${OUTPUT_ROOT}"
     fi

#  save log files produced within 48 hours (in case of long jobs) to tar ball
     echo -e ">>>Making tar ball for log files...\n" 
     find . -mtime -2 -name "*.log" | grep -v "${OUTPUT_LOG%%_[0-9]*.log}\|env_" | xargs tar cvfz ${OUTPUT_FCL//.fcl/.tgz} | tee -a $ENVLOG 
     echo | tee -a $ENVLOG
fi

}

##########################################################################################
# Run the job
##########################################################################################

date -u | tee -a $ENVLOG

STATUS=0
inputfiles=()
count=0
	
if [ $USESAMD -eq 0 ] ; then

# input file or filelist

   if [[ ${INPUT_FILE} != "None" ]]; then
   	 echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -s ${INPUT_FILE}\n" | tee -a $ENVLOG 
   	 $TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -s "${INPUT_FILE}" 2>&1 | tee -a $ENVLOG


   elif [[ "${INPUT_FILELIST}" != "None" ]]; then

         if [ ${ONE_BY_ONE} == "YES" ]; then
      	      echo -e "\n>>>Producing one output file (set) per input file ..." | tee -a $ENVLOG
              for inputfile in `cat ${INPUT_FILELIST}`; do
                  ((count++))

                  if [[ "$inputfile" != *${SCHEMAS}* ]] || [[ -z $SCHEMAS ]]; then
                     #  not schemasified? i.e. no xroot://, no streaming, need to ifdh cp to local area 
                        echo -e "\n>>>ifdh cp -D $inputfile $PWD" | tee -a $ENVLOG
                        if ifdh cp -D $inputfile .; then
                           inputfile=$(basename $inputfile)
                        else
                           echo -e "\n!!!Unable to copy over $inputfile, skipping this file ...\n" | tee -a $ENVLOG
                           date -u | tee -a $ENVLOG
                           ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
                           continue
                        fi
                  fi

                  if [ $count -gt 1 ]; then
                        if ! get_RUNINFO $inputfile ${ONE_BY_ONE}; then
                             echo -e "\n!!!Unable to get RUNINFO for $inputfile, skipping this file ...\n" | tee -a $ENVLOG
                             continue
                        fi
                        ENVLOG="env_${CLUSTER}_run${RUN}.${SUBRUN}.log"
                        rm -f $ENVLOG
                        print_RUNINFO
                        if [[ $OLDRUN == $RUN.$SUBRUN ]]; then
                              ENVLOG="env_${CLUSTER}_run${RUN}.${SUBRUN}.$count.log"
	        	      echo -e "\n!!!Duplicated run info '$OLDRUN' from file '$inputfile', skipping this file ...\n" | tee -a $ENVLOG
                              ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
          	              continue
	                fi
                        sed "s/$OLDRUN/$RUN.$SUBRUN/g" -i ${OUTPUT_FCL}
                        mv ${OUTPUT_FCL} ${OUTPUT_FCL//$OLDRUN/$RUN.$SUBRUN}
                        update_RUNOUTPUT
                  fi

	          date -u | tee -a $ENVLOG

        	  echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -s \"$inputfile\"\n" | tee -a $ENVLOG 
   	          $TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -s "$inputfile" 2>&1 | tee -a $ENVLOG

                  STATUS=${PIPESTATUS[0]}

                  if [[ $STATUS -eq 0 ]]; then
                  #  job success
                        echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, success!\n" | tee -a $ENVLOG

                        RUN_FCLS ${MULTIPLE_FCL}

                  #  job fail
                  else
                        echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, fail!\n" | tee -a $ENVLOG
                  fi

                  ( [[ -z $SCHEMAS ]] || [[ "$inputfile" != *${SCHEMAS}* ]] ) && rm -f "$inputfile"

                  date -u | tee -a $ENVLOG
                  copy_RUNOUTPUT
                  date -u | tee -a $ENVLOG  
                  copy_ENVOUTPUT

              done

         else    
      	      echo -e "\n>>>Producing one output file (set) per job ..." | tee -a $ENVLOG

              for inputfile in `cat "${INPUT_FILELIST}"`; do
                  if [[ "$inputfile" != *${SCHEMAS}* ]] || [[ -z $SCHEMAS ]]; then
                        echo -e "\n>>>ifdh cp -D $inputfile $PWD"   | tee -a $ENVLOG
                        if ifdh cp -D $inputfile .; then
                           inputfiles+=("$(basename $inputfile)")
                        else
                           echo -e "\n!!!Unable to copy over $inputfile, skipping this file ...\n" | tee -a $ENVLOG
                        fi
                  fi
              done
              date -u | tee -a $ENVLOG


              if [ ${#inputfiles[@]} -gt 0 ]; then
   	           echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -s ${inputfiles[@]}\n" | tee -a $ENVLOG 
   	           $TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -s "${inputfiles[@]}" 2>&1 | tee -a $ENVLOG
              else
       	           echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -S ${INPUT_FILELIST}\n" | tee -a $ENVLOG 
   	           $TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -S "${INPUT_FILELIST}" 2>&1 | tee -a $ENVLOG
              fi

         fi #end of ONE_BY_ONE

   else

# no input: MC generation etc.

   	echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT}\n" | tee -a $ENVLOG 
   	$TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} 2>&1 | tee -a $ENVLOG

   fi #end of input_file | input_filelist

   if [[ ${ONE_BY_ONE} == "NO" ]] || [[ ${INPUT_FILE} != "None" ]]; then

        STATUS=${PIPESTATUS[0]}

        if [[ $STATUS -eq 0 ]]; then
        #  job success
           echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, success!\n" | tee -a $ENVLOG

           RUN_FCLS ${MULTIPLE_FCL}

        #  job fail
        else
           echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, fail!\n" | tee -a $ENVLOG
        fi

        date -u | tee -a $ENVLOG 
        copy_RUNOUTPUT
        date -u | tee -a $ENVLOG

        [ ${#inputfiles[@]} -gt 0 ] && rm -f "${inputfiles[@]}"

   fi

else

# sam dataset

   if [ ${IFDH_ART} -eq 1 ]; then
   #  execute job with ifdh_art
      echo -e "\n>>>Producing one output file per job after consuming N input file(s) from ${SAM_PROJECT_URL} with process ID=${SAM_PROCESS_ID}...\n" | tee -a $ENVLOG
      echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} --sam-web-uri=${SAM_PROJECT_URL} --sam-process-id=${SAM_PROCESS_ID}\n" | tee -a $ENVLOG 
      $TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} --sam-web-uri=${SAM_PROJECT_URL} --sam-process-id=${SAM_PROCESS_ID} 2>&1 | tee -a $ENVLOG 

      STATUS=${PIPESTATUS[0]}

      if [[ $STATUS -eq 0 ]]; then
      #  job success
         echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, success!\n" | tee -a $ENVLOG

         RUN_FCLS ${MULTIPLE_FCL}

      else
      #  job fail
         echo -e "\n>>>${OUTPUT_FCL} Command exit status: $STATUS, fail!\n" | tee -a $ENVLOG
      fi

      date -u | tee -a $ENVLOG
      copy_RUNOUTPUT
      date -u | tee -a $ENVLOG


   else
      #  execute jobs without ifdh_art if see incompatability between ifdh_art and ifdhc, or jobs require multiple output files (one per input file)
      #  instead of using sam-art interface for managing the files, handling the metadata, etc., do all these things manually 

      declare -a filenames=()

      echo -e "\n>>>Running gm2 art program without ifdh_art ..." | tee -a $ENVLOG

      if [ ${ONE_BY_ONE} == "YES" ]; then
      	echo -e "\n>>>Producing one output file (set) per input file ..." | tee -a $ENVLOG
      else    
      	echo -e "\n>>>Producing one output file (set) per job ..." | tee -a $ENVLOG
      fi


      while [[ ! -z ${FILE_URI} ]]; do

          ((count++))
          date -u | tee -a $ENVLOG

          if [ $count -gt $((SAMDATASET_MAX_FILES+1)) ]; then
               echo -e "\n!!!Opened too many files ($count > $((SAMDATASET_MAX_FILES+1))) with input $INPUT, stuck in a loop for too long? breaking out...\n" | tee -a $ENVLOG
               STATUS=-1
               break
          elif [ $count -gt 1 ]; then
                echo -e "\n>>>ifdh getNextFile ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}\n" | tee -a $ENVLOG
            	FILE_URI=`ifdh getNextFile ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}`

     		if [[ ! -z ${FILE_URI} ]]; then 
                      if ! get_RUNINFO ${FILE_URI} YES; then
                           echo -e "\n!!!Unable to get RUNINFO for ${FILE_URI}, skipping this file ...\n" | tee -a $ENVLOG
                           continue
                      fi

                      if [ ${ONE_BY_ONE} == "YES" ]; then
                           ENVLOG="env_${CLUSTER}_run${RUN}.${SUBRUN}.log"
                           rm -f $ENVLOG
                      fi

      	              echo -e "\n>>>Next FILE URI to parse: $FILE_URI\n" | tee -a $ENVLOG 
                      print_RUNINFO
                      if [[ $OLDRUN == $RUN.$SUBRUN ]]; then
                            ENVLOG="env_${CLUSTER}_run${RUN}.${SUBRUN}.$count.log"
	        	    echo -e "\n!!!Duplicated run info '$OLDRUN' from '${FILE_URI}', skipping this file ...\n" | tee -a $ENVLOG
                            ifdh cp -D $ENVLOG  ${OUTPUT_DIR}/logs
                            continue
		      fi
                      if [ ${ONE_BY_ONE} == "YES" ]; then
                           sed "s/$OLDRUN/$RUN.$SUBRUN/g" -i ${OUTPUT_FCL}
                           mv ${OUTPUT_FCL} ${OUTPUT_FCL//$OLDRUN/$RUN.$SUBRUN}
                           update_RUNOUTPUT
                      fi
                else
                      echo -e "\n>>>End of FILE_URI list for ${SAM_PROJECT_URL} ${SAM_PROCESS_ID}\n" | tee -a $ENVLOG
                      break
                fi
          fi

       	  if [[ ! -z $SCHEMAS ]] && [[ $FILE_URI == *${SCHEMAS}* ]]; then
                echo -e "\n>>>Found file URI with "${SCHEMAS}", URI = ${FILE_URI}" | tee -a $ENVLOG
	  	filename=$FILE_URI
          else
          # set TMPDIR to local directory to avoid clogging /var/tmp
                TMP_DIR=$TMPDIR
                export TMPDIR=$PWD
      	   	echo -e "\n>>>ifdh fetchInput ${FILE_URI}" | tee -a $ENVLOG
      	   	if filename=`ifdh fetchInput ${FILE_URI}`; then
                   date -u | tee -a $ENVLOG
      	   	   echo -e "\n>>>Fetched file ${filename}" | tee -a $ENVLOG
                else
                   echo -e "\n!!!Unable to fetchInput from ${FILE_URI}, skipping this file ...\n" | tee -a $ENVLOG
                   continue
                fi
                export TMPDIR=${TMP_DIR}

                [[ -z $filename ]] && continue
          fi


          # one output set per input file?
          if [ ${ONE_BY_ONE} == "YES" ]; then
               	echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -s \"${filename}\"" | tee -a $ENVLOG
          	$TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -s "${filename}" 2>&1 | tee -a $ENVLOG

          	STATUS=${PIPESTATUS[0]}

          	if [ $STATUS -eq 0 ]; then
                # job success 
          		echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, success!\n"  | tee -a $ENVLOG

                        RUN_FCLS ${MULTIPLE_FCL}
                    
                        if [ $STATUS -eq 0 ]; then
          	             echo -e "\n>>>Updating the file status to be 'consumed'" | tee -a $ENVLOG
      	   	             echo -e "\n>>>ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename consumed" | tee -a $ENVLOG
      	   	             ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename consumed 2>&1 | tee -a $ENVLOG
                        else
          		     echo -e "\n>>>Updating the file status to be 'skipped'" | tee -a $ENVLOG
          		     echo -e "\n>>>ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped" | tee -a $ENVLOG
          		     ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped 2>&1 | tee -a $ENVLOG
                        fi

		else
                # job fail
          		echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, fail!\n"    | tee -a $ENVLOG
          		echo -e ">>>Updating the file status to be 'skipped'\n" | tee -a $ENVLOG
          		echo -e ">>>ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped" | tee -a $ENVLOG
          		ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped 2>&1 | tee -a $ENVLOG
          	fi

          	( [[ -z $SCHEMAS ]] || [[ ${FILE_URI} != *${SCHEMAS}* ]] ) && rm -f "$filename"

     	        date -u | tee -a $ENVLOG 
     	  	copy_RUNOUTPUT
     	        date -u | tee -a $ENVLOG 
                copy_ENVOUTPUT

	  	[[ ${SAM_FILE_LIMIT} -eq 1 ]] && break

          else

          # single output set per job with multiple input files, run commands later (outside of 'while' loop)

                filenames+=("$filename")
          # unfortunately getNextFile won't work with other status other than 'consumed' or 'skipped', have to set the status as 'consumed' for now 
          # and deal with the consequence later (reset process status if necessary, there is currently NO way to rewrite file status to be 'skipped')
      	   	echo -e "\n>>>ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename consumed" | tee -a $ENVLOG
      	   	ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename consumed 2>&1 | tee -a $ENVLOG

          fi

      done # end of while loop

      date -u | tee -a $ENVLOG      	 

      if [ ${ONE_BY_ONE} == "NO" ]; then
     	echo -e "\n>>>Running command: $TASKSET gm2 $TRACE $TIMING -c ${OUTPUT_FCL} -n ${NEVTSPERJOB} -e ${STARTEVENT} -s ${filenames[@]}\n" | tee -a $ENVLOG
      	$TASKSET gm2 $TRACE $TIMING -c "${OUTPUT_FCL}" -n ${NEVTSPERJOB} -e ${STARTEVENT} -s "${filenames[@]}" 2>&1 | tee -a $ENVLOG

        STATUS=${PIPESTATUS[0]}

        if [ $STATUS -eq 0 ]; then
        #  job success
             echo -e "\n>>>${OUTPUT_FCL} command exit status: $STATUS, success!\n" | tee -a $ENVLOG

             RUN_FCLS ${MULTIPLE_FCL}

        else
        #  job fail
             echo -e "\n>>>${OUTPUT_FCL} Command exit status: $STATUS, fail!\n" | tee -a $ENVLOG

        #  Both the 'consumed' and 'skipped' states are considered to be terminal states by SAM. 
        #  This prevents moving file status from 'consumed' back to 'skipped'
        #  So be it (use process status instead) 
             #echo -e ">>>Updating the file(s) status to be 'skipped'" | tee -a $ENVLOG
             #for filename in ${filenames[@]}; do
             #    echo -e "\n>>>ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped" | tee -a $ENVLOG
             #	 ifdh updateFileStatus ${SAM_PROJECT_URL} ${SAM_PROCESS_ID} $filename skipped 2>&1 | tee -a $ENVLOG
             #done
        fi

        ( [[ -z $SCHEMAS ]] || [[ $FILE_URI != *${SCHEMAS}* ]] ) && rm -f "${filenames[@]}"

        date -u | tee -a $ENVLOG
        copy_RUNOUTPUT
        date -u | tee -a $ENVLOG

      fi # end of $ONE_BY_ONE
  
   fi # end of $IFDH_ART 

fi # end of $USESAMD

# if running locally then the owner must endProject, otherwise jobsub does this for you 
if [ $GRID -eq 0 ]; then    
   ifdh_endPJ ${ONE_BY_ONE}
else
# set status and end SAM project process
   ifdh_endPS ${ONE_BY_ONE}
fi

##################################
# cleanup if needed
##################################
if [ $GRID -eq 0 ] ; then
   rm -f gmon.out
fi

echo -e "\n>>>JOB END with exit status $STATUS\n" 
exit $STATUS
