#!/bin/sh

#please, no sourcing!
[[ $0 != $BASH_SOURCE ]] && echo "Please do not source the script, run it by $BASH_SOURCE" && return

#make sure that the helper scripts exists and are executable
[ ! -r $PWD/submitGM2Data.sh ] && echo "Unable to access submitGM2Data.sh at $PWD, please run from your release area!" && exit
[ ! -r $PWD/GM2DataParserUtils.py ] && echo "Unable to access GM2DataParserUtils.py at $PWD, please run from your release area!" && exit
[ ! -w $PWD ] && echo "Unable to write to current directory at $PWD, please fix it!" && exit 

#check environment setup
[ -z ${MRB_TOP} ] && echo -e "\nDid you run local source setup yet (NOT \". mrb s\")?\n" && exit 1
[ -z ${MRB_PROJECT} ] && echo -e "\nDid you run local source setup yet (NOT \". mrb s\")?\n" && exit 1

#echo "PWD=$PWD"

#help link
HELP="https://cdcvs.fnal.gov/redmine/projects/g-2/wiki/Job_Running_Submission_FAQ"
WIKI="https://cdcvs.fnal.gov/redmine/projects/g-2/wiki/Data_Production"

#FTS directory
FTS_DIR=/pnfs/GM2/scratch/daq

#kinit?
if ! klist -s; then
    echo "kerberos ticket not valid; please run 'kinit'!"
    exit 1
fi

#get the number of input arguments
nargs=$#
if [ $nargs -eq 0 ]; then
   echo "Please type --help for the Help menu, which will provide the list of options."
   exit 0
fi 

#get the input arguments 
list=$@

#determine if help menu is requested  
if [[ $1 == *help* || $1 == "-h" ]] ; then
   python $PWD/GM2DataParserUtils.py $list
   echo " "
   exit 0
fi

#execute the parser and get the variables
opts=`python $PWD/GM2DataParserUtils.py $list`

if [ $? -ne 0 ]; then
     echo -e "\nGM2DataParserUtils.py returned with an error.\n"
     exit 1
fi

#make temporary directory
TMPDIR=`mktemp -d` || 
if [ $? -ne 0 ]; then
     echo -e "\n!!!Unable to create temporary directory using 'mktemp'!!!\n"
     exit 1
fi

# set exit status to be the last non-zero status (if there is any non-zero)
set -o pipefail

#production environment variables
NJOBS=-1
MAX_NJOBS=-1
NEVTSPERJOB=-1
SAM_FILE_LIMIT=""
SCHEMAS="xroot"
REQUEST="9999"

USELOCAL=0
NEWLOCAL=1
OFFSITE=1
OFFSITE_ONLY=0
SITE="None"
BLACKLIST="None"
SITE_OPT=""

GRID=1
DEBUG=0
TEST=0
CLEAN=0
RUNNING_DIR=$PWD
IFDH_ART=1

MEMORY=2000 #default is 2000 MB memory
CPU=1       #default is 1 cpu
DISK=33     #default is about 35000000 KB disk space
LIFETIME=8h #default is 8 hours running time

ROLE="Analysis"
OS="SL6"

MAINFCLNAME="None"
RMAINFCLNAME="None"
FCLPATHNAME="None"
RFCLNAME="None"
FCLNAMES=()
FCLPATHNAMELIST=()
FCLNAMELIST=()
RFCLNAMELIST=()
CAMPAIGN="run"
PROCESS_NAME="None"

INPUT_FILE="None"
INPUT_FILELIST="None"
OUTPUT_DIR="None"

#One output file (set) per input file? 
ONE_BY_ONE="NO"

#single root file? (one output set could already contain multiple root files)
MULTIPLE_ROOT="NO"

#copy json file?
JSON="NO"

DATA_TYPE="None"
DATA_TIER="None"
INCLUDE_ANA=0

RUNNUMBER=1

#sam options
USESAMD=0
USEDATASET=1
SAMDATASET=""
SAMDATASET_OPT=""
PROJECTNAME_OPT=""
SAM_MONITOR_URL="http://samweb.fnal.gov:8480/station_monitor/gm2/stations/gm2/projects/"
SAM_PROJECT_URL="http://samwebgpvm03.fnal.gov:8480/sam/gm2/stations/gm2/projects/name/"


#jobsub options
TARNAME=""
TARFILE=""
TARFILE_OPT=""
INPUT_OPT=""
FCL_OPT=""
TIMEOUT_OPT=""
MAXCONCURRENT_OPT=""
LIFETIME_OPT=""
LINES=""
LINES_OPT1=""
LINES_OPT2=""
CVMFS=0
CVMFS_OPT=""
SL6=0
NOVA=0
NOVA_OPT=""
SUBGROUP_OPT=""

OUTPUT_PNFS_DIR=""

#jobsub status
SUBMITTED=0

#submission log
LOG="$TMPDIR/submission.log"
rm -f $LOG

#parse the list
args=($opts)
for idx in ${!args[@]};
do
   key="${args[idx]}"
   value="${args[((idx+1))]}"

   #echo "key=$key"     | tee -a $LOG
   #echo "value=$value" | tee -a $LOG

   # basic variables
   if [[ $key == "njobs" ]] ; then
      NJOBS="$value"
   fi

   if [[ $key == "maxconcurrent" ]] ; then
      MAX_NJOBS="$value"
   fi

   if [[ $key == "nevents" ]] ; then
      NEVTSPERJOB="$value"
   fi 

   if [[ $key == "runNumber" ]] ; then
      RUNNUMBER="$value"
   fi 

   if [[ $key == "campaign" ]] ; then
      CAMPAIGN="$value"
   fi

   if [[ $key == "process" ]] ; then
      PROCESS_NAME="$value"
   fi

   if [[ $key == "subgroup" ]] ; then
      [[ $value != "None" ]] && SUBGROUP_OPT="--subgroup $value"
   fi

   if [[ $key == "localArea" && $value == "True" ]] ; then
      USELOCAL=1
   fi

   if [[ $key == "samelocal" && $value == "True" ]] ; then
      NEWLOCAL=0
   fi


   # production only
   if [[ $key == "useProRole" && $value == "True" ]] ; then
      ROLE="Production"
   fi

   # production only
   if [[ $key == "requestid" ]] ; then
      REQUEST="$value"
   fi

   if [[ $key == "memory" ]] ; then
      MEMORY=$value
   fi

   if [[ $key == "cpu" ]] ; then
      CPU=$value
   fi

   if [[ $key == "lifetime" ]] ; then
      LIFETIME=$value
   fi

   if [[ $key == "timeout" ]] ; then
      TIMEOUT=$value
   fi

   if [[ $key == "disk" ]] ; then
      DISK=$value
   fi

   if [[ $key == "offsite" ]] ; then
      OFFSITE=$value
   fi

   if [[ $key == "offsite-only" && $value == "True" ]] ; then
      OFFSITE_ONLY=1
   fi

   if [[ $key == "cvmfs" && $value == "True" ]] ; then
      CVMFS=1
   fi

   if [[ $key == "nova" && $value == "True" ]] ; then
      NOVA=1
   fi

   if [[ $key == "sl6" && $value == "True" ]] ; then
      SL6=1
   fi

   if [[ $key == "site" && $value != "None" ]] ; then
      SITE="$value"
   fi

   if [[ $key == "blacklist" && $value != "None" ]] ; then
      BLACKLIST="$value"
   fi

   if [[ $key == "noGrid" && $value == "True" ]] ; then
      GRID=0
   fi

   if [[ $key == "test" && $value == "True" ]] ; then
      TEST=1
   fi

   if [[ $key == "cleanup" && $value == "True" ]] ; then
      CLEAN=1
   fi

   if [[ $key == "debug" && $value == "True" ]] ; then
      DEBUG=1
   fi

   if [[ $key == "toplocaldir" && $value == "True" ]] ; then
      RUNNING_DIR=${MRB_TOP}/run
      mkdir -p ${RUNNING_DIR}
      if [ $? -ne 0 ]; then
           echo -e "\n!!!Unable to create local running directory at ${RUNNING_DIR}!!!\n"
           exit 1
      fi 
      echo -e "\nChanging local running directory to ${RUNNING_DIR} ...\n"
      ln -sf $PWD/submitGM2Data.sh ${RUNNING_DIR} 
      cd ${RUNNING_DIR}
   fi

   if [[ $key == "noifdh_art" && $value == "True" ]] ; then
      IFDH_ART=0
   fi

   if [[ $key == "input-file" && $value != "None" ]] ; then
      INPUT_FILE="$value"
   fi

   if [[ $key == "input-filelist" && $value != "None" ]] ; then
      INPUT_FILELIST="$value"
   fi

   if [[ $key == "output-dir" && $value != "None" ]] ; then
      OUTPUT_DIR="$value"
   fi

   # one output file (set) per input file
   if [[ $key == "onebyone" && $value == "True" ]] ; then
      ONE_BY_ONE="YES"
   fi

   if [[ $key == "multipleroot" && $value == "True" ]] ; then
      MULTIPLE_ROOT="YES"
   fi


   if [[ $key == "json" && $value == "True" ]] ; then
      JSON="YES"
   fi

   if [[ $key == "lines" && $value != "None" ]] ; then
      LINES="$value"
   fi

   if [[ $key == "schemas" ]] ; then
      SCHEMAS="$value"
   fi

   if [[ $key == "schemas_uri" ]] ; then
      SCHEMAS_URI="$value"
   fi


   # get data type
   if [[ $key == "daq" && $value == "True" ]] ; then
      DATA_TYPE="$key"
   elif [[ $key == "mc" && $value == "True" ]] ; then
      DATA_TYPE="$key"
   fi


   # get fhicl file
   if [[ $key == "fhiclFile" && $value != "None" ]] ; then
      MAINFCLNAME="$value"
      set -f   #disable glob
      IFS=','  #comma,separated,list
      FCLNAMES=($MAINFCLNAME)
      unset IFS
      set +f
   fi

   # sam variables
   if [[ $key == "sam-dataset" ]] ; then
      SAMDATASET="$value"
      [[ $SAMDATASET != "None" ]] && USESAMD=1
   fi

   if [[ $key == "sam-max-files" ]] ; then
      SAM_FILE_LIMIT="$value"
   fi
 

   # get data tier
   if [[ $key == "truth" && $value == "True" ]] ; then
      DATA_TIER="$key"
   elif [[ $key == "unpack" && $value == "True" ]] ; then
      DATA_TIER="$key"
   elif [[ $key == "digit" && $value == "True" ]] ; then
      DATA_TIER="$key"
   elif [[ $key == "reco" && $value == "True" ]] ; then
      DATA_TIER="$key"
   elif [[ $key == "full" && $value == "True" ]] ; then
      DATA_TIER="$key"
   elif [[ $key == "ana" && $value == "True" ]] ; then
      DATA_TIER="$key"
      INCLUDE_ANA="1"
   fi

done

# no need to use dataset for truth simulation
if [[ ${DATA_TIER} == "truth" ]] ; then
   USEDATASET=0
fi 

if [[ $NJOBS -lt 1 ]]; then
   echo -e "\n!!!Number of jobs missing or invalid!!!\n"
   exit 1
fi

if [[ ${INPUT_FILE} != "None" ]] || [[ ${INPUT_FILELIST} != "None" ]]; then

   USEDATASET=0
   if [ $USESAMD -eq 1 ]; then
   	echo -e "\n!!!Cannot use SAM dataset and input-file/input-filelist options at the same time!!!\n"
   	exit 1
   fi

   if [ ${IFDH_ART} -eq 1 ]; then
   	echo -e "\n!!!IFDH_ART interface not supported for input-file/input-filelist options, use --noifdh_art instead!!!\n"
   	exit 1
   fi

   if [[ $GRID -eq 1 ]] && [[ $NJOBS -gt 1 ]] && [[ ${INPUT_FILE} != "None" ]]; then
      echo -e "\n***You are submitting $NJOBS jobs with a single input file." 
      echo -e "***Each job will consume the same input file, i.e. the same input will be run $NJOBS times."
      echo -e "***If this is NOT what is intended, consider submitting a single job or using a filelist or SAM dataset instead." 
      echo -e "***If you wish to change the configuration and re-submit, Ctrl+C to exit now!\n"
      sleep 7
   fi

fi 

if [[ ${INPUT_FILE} != "None" ]] && [[ ${INPUT_FILELIST} != "None" ]]; then
   	echo -e "\n!!!Cannot use input file and input filelist at the same time!!!\n"
   	exit 1
fi


#make sure the production processing data tier is defined
if [ ${DATA_TIER} == "None" ]; then
   if [ ${INCLUDE_ANA} == "1" ] ; then
      	DATA_TIER="ana"
   elif [ ${DATA_TYPE} == "mc" ]; then
        DATA_TIER="full"
   else 
     	echo -e "\nData tier is NOT defined!\n"
     	exit 1
   fi
fi

if [[ ${MAINFCLNAME} == "None" ]] ; then
   echo -e "\nFHiCL file is NOT defined!\n"
   exit 1
fi

#Production mode
[[ $USER == "gm2pro" ]] && ROLE="Production"
  
if [[ $ROLE == "Production" ]] || [[ ${OUTPUT_DIR} == *${FTS_DIR}/* ]]; then
#make sure production has a request id
      if [[ $REQUEST == "9999" ]] && [[ $DEBUG -eq 0 ]]; then
            echo -e "\n!!!Request ID is missing for production jobs!!!\n"
            exit 1
      fi
#make sure CAMPAIGN is set properly if SAMDATASET or REQUESTID is defined
      [[ $SAMDATASET == *run1* ]] && CAMPAIGN=run1
      [[ $SAMDATASET == *run2* ]] && CAMPAIGN=run2
      [[ $SAMDATASET == *run3* ]] && CAMPAIGN=run3
      [[ $SAMDATASET == *run4* ]] && CAMPAIGN=run4
      [[ $REQUEST == 50*       ]] && CAMPAIGN=run1
      [[ $REQUEST == 51*       ]] && CAMPAIGN=run2
      [[ $REQUEST == 52*       ]] && CAMPAIGN=run3
      [[ $REQUEST == 53*       ]] && CAMPAIGN=run4

#make sure maxconcurrent is set to be 10000
      MAXCONCURRENT_OPT="--maxConcurrent 10000"

#make sure job subgroup is set to be 'prod' by default
      [[ -z ${SUBGROUP_OPT} ]] && SUBGROUP_OPT="--subgroup prod"
#make sure simulation production jobs are assigned to 'sim' subgroup
      [[ ${DATA_TYPE} == "mc" ]] && SUBGROUP_OPT="--subgroup sim"
fi


if [[ $GRID -eq 0 && ! -z ${mrb_command} ]]; then 
      echo -e "\n***You are running jobs locally, no need to setup mrb environment (NO \". mrb s\"). The script does everything for you."
      echo -e "***Proceed at your own risk or restart with a new shell and run again.***" 
      sleep 5
fi

#####################################################################
#The output job information
#####################################################################
ln -sf $LOG

echo -e "$PWD\n\n" >> $LOG
echo $0 $list | tee -a $LOG
#echo $0 $opts | tee -a $LOG
echo -e "\nSubmission log: $LOG\n" 
if [[ ${DATA_TYPE} == "mc" ]] ; then
   echo -e "\n------ Running MC campaign: \"${CAMPAIGN}\", processing \"${DATA_TIER}\" data tier ($MRB_PROJECT_VERSION) -----------" | tee -a $LOG
elif [[ ${DATA_TYPE} == "daq" ]] ; then
   echo -e "\n------ Running DAQ campaign: \"${CAMPAIGN}\", processing \"${DATA_TIER}\" data tier ($MRB_PROJECT_VERSION) -----------" | tee -a $LOG
else
   echo -e "\n!!!Invalid DATA TYPE: ${DATA_TYPE}!!!\n"
   exit 1
fi

##################################
# set the sam data set option
##################################
if [ $USEDATASET -eq 1 ] ; then
   if [ $USESAMD -eq 0 ] ; then
      echo -e "\n!!!Unable to submit and process the dataset, --sam-dataset is missing!!!\n" | tee -a $LOG
      exit 1
   else
      echo -e "\nProcessing dataset name: ${SAMDATASET} ..." | tee -a $LOG
   fi

   SAMDATASET_MAX_FILES=`samweb count-definition-files ${SAMDATASET}`

   if [[ -z $SAMDATASET_MAX_FILES ]] || [[ $SAMDATASET_MAX_FILES -le 0 ]]; then
      echo -e "\n!!!SAM dataset ${SAMDATASET} is invalid or contains 0 files!!!\n" | tee -a $LOG
      exit 1
   fi

   if [[ ! -z $SAM_FILE_LIMIT ]] && [[ $SAM_FILE_LIMIT -gt 0 ]] && [[ $SAM_FILE_LIMIT -lt $SAMDATASET_MAX_FILES ]] ; then
      echo -e "\n***'sam-max-files' is set to be $SAM_FILE_LIMIT, which is less than the total number of files (${SAMDATASET_MAX_FILES}) from ${SAMDATASET} dataset." | tee -a $LOG
      echo "***This will cause each job to (randomly) select and consume only $SAM_FILE_LIMIT file(s) from ${SAMDATASET} dataset." | tee -a $LOG
      echo "***It is OK to do this if you are running a test or you are merging many input files into one output file," | tee -a $LOG
      echo "***although this might hurt the overall job efficiency." | tee -a $LOG
      echo "***The recommended way to run over all files from ${SAMDATASET} dataset is to NOT set 'sam-max-files' at all." | tee -a $LOG
      echo "***This will allow all ${SAMDATASET_MAX_FILES} files to be (randomly and more or less evenly) consumed by N jobs and produce one output file per job." | tee -a $LOG
      echo "***Note that if you require one output file per input file, please also use --noifdh_art and --onebyone switches." | tee -a $LOG 
      echo -e "***If you wish to change the configuration and re-submit, Ctrl+C to exit now!\n" | tee -a $LOG
      sleep 10
   fi

   PROJECTNAME=${USER}_`date +%Y%m%d%H`_$$
   SAMDATASET_OPT="--dataset_definition=$SAMDATASET -e SAMDATASET=$SAMDATASET -e SAMDATASET_MAX_FILES=${SAMDATASET_MAX_FILES} -e SAM_FILE_LIMIT=${SAM_FILE_LIMIT}"
   PROJECTNAME_OPT="--project_name=${PROJECTNAME}"
   SAM_MONITOR_URL="${SAM_MONITOR_URL}$PROJECTNAME"
   SAM_PROJECT_URL="${SAM_PROJECT_URL}$PROJECTNAME"

fi    


#####################################################################
#Get a VOMS proxy if necessary
#####################################################################

if [[ $SCHEMAS == "None" ]] || [[ $SCHEMAS == "none" ]]; then
      SCHEMAS=""
fi

if [[ "${INPUT_FILE}" == xroot* ]] && [[ $SCHEMAS != xroot ]]; then
      echo -e "\n!!!Input-file '${INPUT_FILE}' starts with xroot but SCHEMAS is NOT set as xroot, please fix this!!!\n"
      exit 1
fi

# currently there is no streaming support for non-root files ("Unpacking"), also there is no input needed for "Truth" generation 
if [[ ${DATA_TIER} == "unpack" ]] || [[ ${DATA_TIER} == "truth" ]] || [ ${DATA_TIER} == "full" -a ${DATA_TYPE} == "daq" ]; then
        SCHEMAS=""
elif [[ ! -z $SCHEMAS ]]; then
# voms-proxy command only needed for users (see below for production jobs)
        if [[ $ROLE == "Analysis" ]]; then
              echo | tee -a $LOG
              voms-proxy-destroy > /dev/null 2>&1
              if kx509 | tee -a $LOG; then
                   if voms-proxy-init -rfc -noregen -voms=fermilab:/fermilab/gm2/Role=$ROLE 2>&1 | tee -a $LOG; then
# voms-proxy-init error codes are out of whack, no reason to trust them before a fix from OSG team
                        sleep 1
                        if [ $(voms-proxy-info -all | grep -c "VO fermilab") -eq 0 ]; then
                             echo -e "\n***voms-proxy reported a problem or warning, please check above printout for detailed information."
                             echo "***If you choose to continue, do nothing. Jobs will be submitted with SCHMEAS set to be \"$SCHEMAS\" regardless of any reported issue."
                             echo "***You can also choose to press Ctrl+C to exit now and try again later (from a different machine)."
                             sleep 10
                        fi
                   else
                        echo -e "\n***Unable to run 'voms-proxy-init -rfc -noregen -voms=fermilab:/fermilab/gm2/Role=$ROLE', set SCHEMAS to be 'None' in order to continue."
                        echo -e "***You can also choose to press Ctrl+C to exit now and try again later (from a different machine)."
                        SCHEMAS=""
                        sleep 10
                   fi
              else
                   echo -e "\n!!!Unable to get a valid certificate, re-run 'kinit' and/or 'kx509' (from a different machine) and check for errors!!!\n"
                   exit 1
              fi
        else
              export X509_USER_PROXY=/opt/gm2pro/gm2pro.Production.proxy
        fi
fi

#####################################################################
#Default URI="fndca1.fnal.gov/pnfs/fnal.gov/usr/"
SCHEMAS_URI="${SCHEMAS}://${SCHEMAS_URI}"
#####################################################################
#####################################################################
#Check input file(s) if necessary
#####################################################################
ifile=0
NINPUT=0
NPERJOB=-1
TMPLIST=$TMPDIR/$(basename ${INPUT_FILELIST})

if [[ $USESAMD -eq 0 ]] && [[ ${DATA_TIER} != "truth" ]]; then
      if [[ ${INPUT_FILE} == "None" ]] && [[ ${INPUT_FILELIST} == "None" ]]; then
            echo -e "\n!!!Input-file or input-filelist must be provided for ${DATA_TIER} tier if NO sam dataset is defined!!!\n" | tee -a $LOG
            exit 1
      elif [[ ${INPUT_FILE} != "None" ]]; then

            if [[ "${INPUT_FILE}" != *${SCHEMAS}* ]] || [[ -z $SCHEMAS ]]; then
            #not schemafied? i.e. no xroot://....
		if [[ ! -f "${INPUT_FILE}" ]] || [[ ! -r "${INPUT_FILE}" ]]; then
        		echo -e "\n!!!Input-file '${INPUT_FILE}' cannot be accessed!!!\n" | tee -a $LOG
        		exit 1
	        elif [[ "${INPUT_FILE}" == */pnfs/gm2/* ]]; then
            	        echo -e "\n!!!Input file '${INPUT_FILE}' has /pnfs/gm2/ in its name (should be /pnfs/GM2/)!!!\n" | tee -a $LOG
                        exit 1
		elif [[ "$INPUT_FILE" == /pnfs/*.root ]] && [[ ! -z $SCHEMAS ]]; then
                	INPUT_FILE=${INPUT_FILE/\/pnfs\//${SCHEMAS_URI}}
                        [ $GRID -eq 1 ] && INPUT_OPT="-f ${INPUT_FILE}"
                elif [[ "$INPUT_FILE" == /pnfs/* ]]; then
                        [ $GRID -eq 1 ] && INPUT_OPT="-f ${INPUT_FILE}"
                else
                        [ $GRID -eq 1 ] && INPUT_OPT="-f dropbox://${INPUT_FILE}"			
		fi
	    else
                if [[ ${INPUT_FILE} == */usr/gm2/* ]]; then
            	        echo -e "\n!!!Input file '${INPUT_FILE}' has /usr/gm2/ in its name (should be /usr/GM2/)!!!\n" | tee -a $LOG
                        exit 1
                else
                        [ $GRID -eq 1 ] && INPUT_OPT="-f ${INPUT_FILE}"
                fi
            fi

      elif [[ ${INPUT_FILELIST} != "None" ]]; then
            if [[ ! -f "${INPUT_FILELIST}" ]] || [[ ! -r "${INPUT_FILELIST}" ]]; then
            	echo -e "\n!!!Input-filelist '${INPUT_FILELIST}' cannot be accessed!!!\n" | tee -a $LOG
       		exit 1
      	    else
                NINPUT=$(wc -l < ${INPUT_FILELIST})
        	if [[ ! -z $SCHEMAS ]]; then
                #schema defined? i.e. xroot...
                 	for file in `cat ${INPUT_FILELIST}`; do
                        	if [[ "$file" == */pnfs/gm2/* ]]; then
                                	echo -e "\n!!!Input file '$file' inside '${INPUT_FILELIST}' has /pnfs/gm2/ in its name (should be /pnfs/GM2/)!!!\n" | tee -a $LOG
                                	exit 1
                        	fi
                        	if [[ "$file" == */usr/gm2/* ]]; then
                                	echo -e "\n!!!Input file '$file' inside '${INPUT_FILELIST}' has /usr/gm2/ in its name (should be /usr/GM2/)!!!\n" | tee -a $LOG
                                	exit 1
                        	fi
                        	if [[ ! -z "$file" ]] && [[ "$file" != *${SCHEMAS}* ]] && [[ ! -f "$file" ]]; then
                                	echo -e "\n!!!Input file '$file' inside '${INPUT_FILELIST}' does NOT exist!!!\n" | tee -a $LOG
                                	exit 1
                        	fi
                        	if [[ "$file" == /pnfs/*.root ]]; then 
                                 	file=${file/\/pnfs\//${SCHEMAS_URI}}
					(( ifile++ ))
				fi
                        	echo $file >> $TMPLIST
                	done
			if [ $ifile -gt 0 ]; then
                		echo -e "\nTransformed $ifile out of $NINPUT files from $INPUT_FILELIST to $TMPLIST with $SCHEMAS" |  tee -a $LOG
	         		INPUT_FILELIST=$TMPLIST
			fi

	         fi # schemas transformation done
            fi #readble INPUT_FILELIST
      fi #given INPUT
fi # sam or inputfile/filelist?



##############################################################################
#The FHiCL file you want to use for the submission
##############################################################################

#Loop over FCL files
for MAINFCLNAME in "${FCLNAMES[@]}"; do
    [[ -z "$MAINFCLNAME" ]] && continue
    if [ ! -r ${MRB_TOP}/srcs/gm2analyses/fcl/${MAINFCLNAME} ] ; then
         if [ ! -r ${MAINFCLNAME} ] ; then
              echo -e "\n!!!Unable to access ${MAINFCLNAME}!!!"  | tee -a $LOG
              echo -e "FHiCL file(s) needs to be in gm2analyses/fcl directory or specified with full path." | tee -a $LOG  
              echo -e "See production wiki $WIKI for examples." | tee -a $LOG 
              echo -e "MRB_TOP=${MRB_TOP}\n" | tee -a $LOG
              exit 1
         else
              FCLPATHNAME=`readlink -m ${MAINFCLNAME}`
         fi
    else
         FCLPATHNAME=`readlink -m ${MRB_TOP}/srcs/gm2analyses/fcl/${MAINFCLNAME}`
    fi

    MAINFCLNAME=$(basename $FCLPATHNAME)
    RMAINFCLNAME=$MAINFCLNAME

    if [ $GRID -eq 0 ]; then
	cp -f ${FCLPATHNAME} ${MAINFCLNAME}.tmp
        if [ $? -ne 0 ]; then
             echo -e "\n!!!Unable to copy ${FCLPATHNAME} to $PWD!!!\n" | tee -a $LOG
             exit 1
        fi 
	RMAINFCLNAME=${MAINFCLNAME}.tmp
    fi

    FCLPATHNAMELIST+=($FCLPATHNAME)
    FCLNAMELIST+=($MAINFCLNAME)
    RFCLNAMELIST+=($RMAINFCLNAME)

done

echo -e "\nMain FCL(s): ${FCLNAMELIST[@]}"           | tee -a $LOG
echo -e "Remote Main FCL(s): ${RFCLNAMELIST[@]}\n"   | tee -a $LOG

IFS=','
eval 'RFCLNAME="${RFCLNAMELIST[*]}"'
unset IFS

if [ $GRID -eq 0 -a $NJOBS -gt 1 ]; then
      echo -e "\n***Submitting $NJOBS jobs with --noGrid option. However multiple jobs running is only possible with --Grid option." | tee -a $LOG 
      echo -e "***Proceed with single job running mode now (i.e. --njobs 1) ...\n" | tee -a $LOG
      NJOBS=1
      sleep 5
fi 

#####################################################################
#IFDH_ART service status
#####################################################################

if [ ${IFDH_ART} -eq 1 ]; then
      echo "IFDH_ART interface activated"

      if [[ ${ONE_BY_ONE} == "YES" ]]; then
            echo -e "\n***IFDH_ART interface is activated, which by default only produces a single output set per job regardless of the number of input files." | tee -a $LOG 
            echo -e "***Proceed with single output set mode now (--onebyone switch will be ignored) ..." | tee -a $LOG 
            echo -e "***If this is NOT what is intended, Ctrl+C to exit now and re-submit with --noifdh_art option!\n" | tee -a $LOG
            sleep 7
      fi     

else
      echo "IFDH_ART interface disabled"
fi

#####################################################################
#How many events/outputs per job and how many jobs
#####################################################################
echo
echo "Number of jobs: $NJOBS"                       | tee -a $LOG
if [ $NINPUT -gt 0 ]; then
     echo "Number of input files: $NINPUT"          | tee -a $LOG
     NPERJOB=$((NINPUT/NJOBS + 1))
     [ $NINPUT -eq $((NJOBS*(NPERJOB-1))) ] && NPERJOB=$((NPERJOB-1))
     echo "Number of input files per job: $NPERJOB" | tee -a $LOG
     [ $NPERJOB -le 0 ] && echo -e "\n!!!Invalid number of input files per job: $NPERJOB <= 0!!!" && exit 1
fi

if [ $USESAMD -eq 1 ]; then
     echo "Number of dataset files: $SAMDATASET_MAX_FILES" | tee -a $LOG
fi

echo "Max number of concurrent jobs: $MAX_NJOBS"    | tee -a $LOG
echo "Number of events per job: $NEVTSPERJOB"       | tee -a $LOG
echo "One output file (set) per input file: $ONE_BY_ONE"  | tee -a $LOG
echo "Multiple root files per output set: $MULTIPLE_ROOT" | tee -a $LOG
echo



######################################################################################
#create a place to put your output. This creates a directory in the /pnfs area
#which is based on your Fermilab user name, and the current date and time. 
#######################################################################################
if [[ "${OUTPUT_DIR}" == "None" ]] ; then
   OUTPUT_DIR=/pnfs/GM2/scratch/users/${USER}/${CAMPAIGN}/${DATA_TYPE}/${DATA_TIER}
fi

if [[ "${OUTPUT_DIR}" == /pnfs/gm2/* ]] ; then 
   echo -e "\n!!!Please use /pnfs/GM2 instead of /pnfs/gm2 in the output directory path!!!\n" | tee -a $LOG
   exit 1
fi
  
NOW=$(date +"%F-%H-%M-%S")
OUTPUT_DIR=${OUTPUT_DIR}/${NOW}

if [[ -e ${OUTPUT_DIR} ]]; then
   echo -e "\n!!!Output directory ${OUTPUT_DIR} already exists!!!\n" | tee -a $LOG
   exit 1
fi 

mkdir -p ${OUTPUT_DIR} 
if [ $? -ne 0 ]; then
     echo -e "\n!!!Unable to make output directory ${OUTPUT_DIR}!!!\n" | tee -a $LOG
     exit 1
fi

mkdir ${OUTPUT_DIR}/logs
mkdir ${OUTPUT_DIR}/data
mkdir ${OUTPUT_DIR}/fcl

if [[ ${JSON} == "YES" ]]; then
   mkdir ${OUTPUT_DIR}/json
fi


chmod -R g+w ${OUTPUT_DIR}

echo -e "\nOutput directory: ${OUTPUT_DIR}" | tee -a $LOG

if [[ ${INPUT_FILELIST} != "None" ]]; then
	echo -e "\nCopy input filelist ${INPUT_FILELIST} to the output directory ..." | tee -a $LOG
	ifdh cp -D ${INPUT_FILELIST} ${OUTPUT_DIR} 
        if [ $? -ne 0 ]; then
           echo -e "\n!!!Unable to copy ${INPUT_FILELIST} to ${OUTPUT_DIR}!!!\n" | tee -a $LOG
           exit 1
        fi
                
        [ $GRID -eq 1 ] && INPUT_OPT="-f dropbox://${OUTPUT_DIR}/$(basename ${INPUT_FILELIST})"
else
	echo -e "\nCopy main FHiCL file to the output directory ..." | tee -a $LOG
fi

for FCLPATHNAME in "${FCLPATHNAMELIST[@]}"; do
    if ! ifdh cp -D "$FCLPATHNAME" "${OUTPUT_DIR}"; then
         echo -e "\n!!!Unable to copy ${FCLPATHNAME} to ${OUTPUT_DIR}!!!\n" | tee -a $LOG 
         exit 1
    fi
    FCL_OPT="-f dropbox://${OUTPUT_DIR}/$(basename $FCLPATHNAME) ${FCL_OPT}"
done

#echo -e "\nFCL_OPT=$FCL_OPT\n"

#############################################
#Tar the local area if requested
#############################################
if [ $USELOCAL -eq 1 ]; then
        echo -e "\nUsing local setup at ${MRB_INSTALL} ..." | tee -a $LOG
      	if [ $(ls -d ${MRB_INSTALL}/* | wc -l) -le 1 ]; then
      		echo -e "\n!!!Did not find any local packages at ${MRB_INSTALL}!!!"    | tee -a $LOG
                echo -e "!!!please do \". mrb s\" and \"mrb b && mrb i\" first!!!\n" | tee -a $LOG 
		exit 1
	fi
	if [ $GRID -eq 1 ]; then

                TARNAME=${MRB_PROJECT}_${MRB_PROJECT_VERSION}_${USER}_$(basename ${MRB_TOP}).tgz
                OUTPUT_PNFS_DIR=/pnfs/GM2/scratch/users/${USER}/localArea
                if [ ! -d ${OUTPUT_PNFS_DIR} ]; then
                        mkdir -p ${OUTPUT_PNFS_DIR}
                        chmod -R g+w ${OUTPUT_PNFS_DIR}
                fi
                if [ ! -d ${OUTPUT_PNFS_DIR} ]; then
                        echo -e "\n\tUnable to make directory at ${OUTPUT_PNFS_DIR}, use $PWD instead ..." | tee -a $LOG
                        TARFILE=$PWD/$TARNAME
                else
                        TARFILE=${OUTPUT_PNFS_DIR}/$TARNAME
                fi

                if [[ -e $TARFILE ]] && [[ $NEWLOCAL -eq 0 ]]; then
                        echo -e "\n\tUsing the existing tar file $TARFILE for submission ..." | tee -a $LOG
                else
                        echo -e "\n\tMaking new tar file $TARFILE for submission ..."         | tee -a $LOG
                        if rm -f $TARFILE; then
                           if tar cfz $TARNAME -C / ${MRB_INSTALL/\//}; then
                              [ -d ${OUTPUT_PNFS_DIR} ] && mv $TARNAME ${OUTPUT_PNFS_DIR}
                           else
                              echo -e "\n!!!Unable to make tar file $PWD/$TARNAME, check /tmp or TMPDIR!!!\n" | tee -a $LOG  && exit 1
                           fi
                        else
                              echo -e "\n!!!Unable to update tar file $TAFILE!!!\n" | tee -a $LOG && exit 1
                        fi

                fi
                [ ! -e $TARFILE ] && echo -e "\n!!!Unable to create $TARFILE!!!\n" | tee -a $LOG  && exit 1


                TARFILE_OPT="--tar_file_name dropbox://$TARFILE"
#               TARFILE_OPT="--tar_file_name tardir://${MRB_INSTALL}"
        else
		echo -e "\nYou are already running jobs locally with --noGrid option. The script will look over ${MRB_INSTALL} and perform necessary setup and then run jobs locally." | tee -a $LOG
        	sleep 3
	fi
fi



####################################################################
#link fetchlogid.sh for user's convenience in case of grid running
####################################################################
if [ $GRID -eq 1 ]; then
	ln -sf $PWD/fetchlog.sh $OUTPUT_DIR
	echo -e "\nRun './fetchlog.sh' or 'sh fetchlog_*.sh' (from the output directory) to fetch job log files\n" | tee -a $LOG
fi

##########################################
#create the run number
##########################################
if [[ ${DATA_TIER} == "truth" ]] ; then 
   RUNNUMBER=`date +%s`
   echo -e "Truth generation starting at run number: $RUNNUMBER\n" | tee -a $LOG
fi

if [[ $RUNNUMBER =~ [^0-9]+ ]]; then
   echo -e "\n!!!Invalid run number: $RUNNUMBER!!!\n"  | tee -a $LOG
   exit 1
fi

#################################
#concatenate arguments
#################################
if [ $GRID -eq 0 ]; then
     INPUTLIST="${opts} runNumber ${RUNNUMBER} remoteFile ${RFCLNAMELIST[0]} remoteFiles $RFCLNAME outfiles ${OUTPUT_DIR} $NJOBS $NPERJOB"
else
     INPUTLIST="${opts} runNumber ${RUNNUMBER} remoteFile ${RFCLNAMELIST[0]} remoteFiles $RFCLNAME outfiles ${OUTPUT_DIR} $NJOBS $NPERJOB"
fi

##################################
#where to submit jobs 
#################################
USAGE_MODEL="DEDICATED,OPPORTUNISTIC"
if [[ $OFFSITE == 1 ]] || [[ $OFFSITE == "Yes" ]] || [[ $OFFSITE == "yes" ]] || [[ $OFFSITE == "True" ]] || [[ $OFFSITE == "true" ]]; then
   USAGE_MODEL="DEDICATED,OPPORTUNISTIC,OFFSITE"
   OFFSITE=1
elif [[ $OFFSITE == -* ]]; then
   echo -e "\n!!!Invalid offsite option: $OFFSITE!!!\n" | tee -a $LOG
   exit 1
elif [[ ${OFFSITE_ONLY} == 1 ]]; then
   USAGE_MODEL="OFFSITE"
fi

##################################
# get the site option
##################################
if [[ ! -z $SITE ]] && [[ $SITE != "None" ]]; then
   SITE_OPT="--site=$SITE"
fi

if [[ ! -z $BLACKLIST ]] && [[ $BLACKLIST != "None" ]]; then
#  temporary fix until --blacklist really works
#  SITE_OPT="--blacklist=$BLACKLIST ${SITE_OPT}"

#  SITE_OPT="--append_condor_requirements='(TARGET.GLIDEIN_Site =!= \"$BLACKLIST\")' ${SITE_OPT}"
   SITE_OPT="--append_condor_requirements='(TARGET.GLIDEIN_Site isnt \"$BLACKLIST\")' ${SITE_OPT}"
fi


##################################
# get the CVMFS option
##################################
if [[ $CVMFS -eq 1 ]]; then
   CVMFS_OPT="--append_condor_requirements='(TARGET.HAS_CVMFS_gm2_opensciencegrid_org==true)'"
fi

if [[ $NOVA -eq 1 ]]; then
   NOVA_OPT="--append_condor_requirements='(TARGET.HAS_CVMFS_nova_opensciencegrid_org==true)'"
fi

##################################
# get the lines option
##################################
if [[ ! -z $LINES ]] && [[ $LINES != "None" ]]; then
   LINES_OPT1=$LINES
fi

if [ $SL6 -eq 1 ]; then
   LINES_OPT2=+SingularityImage=\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl6:latest\"
   OS="CentOS7"
fi

##################################
# get the memory usage
##################################
if [[ $MEMORY -le 100 ]]; then
   MEMORYUSAGE="${MEMORY}GB"
else
   MEMORYUSAGE="${MEMORY}MB"
fi

##################################
# get the disk usage
##################################
DISKUSAGE="${DISK}GB"

##################################
# set the expected-lifetime option
##################################
if [[ ! -z $LIFETIME ]] && [[ $LIFETIME != "8h" ]]; then
   LIFETIME_OPT="--expected-lifetime $LIFETIME"
fi

##################################
# set the timeout option
##################################
if [[ ! -z $TIMEOUT ]] && [[ $TIMEOUT != "-1" ]]; then
   TIMEOUT_OPT="--timeout $TIMEOUT"
fi

##################################
# set the maxconcurrent option
##################################
if [[ ! -z ${MAX_NJOBS} ]] && [[ ${MAX_NJOBS} != "-1" ]]; then
   MAXCONCURRENT_OPT="--maxConcurrent ${MAX_NJOBS}"
fi
            

##################################
# set the tarfile option
# As of April 2019, switching to --tar_file_name option for better performance
##################################
#if [ $USELOCAL -eq 1 ] ; then
#  TARFILE_OPT="-f ${TARFILE}"
#fi


###################################################
#This submits the job to the grid.
###################################################
if [ $GRID -eq 1 ]; then

    CMD="jobsub_submit    \
           ${FCL_OPT}     \
           ${INPUT_OPT}   \
           ${TARFILE_OPT} \
           -N ${NJOBS} -G gm2 -e NJOBS=$NJOBS -e NPERJOB=$NPERJOB -e RFCLNAME=$RFCLNAME -e ROLE=$ROLE -e CAMPAIGN=$CAMPAIGN \
           -e RELEASE=${MRB_PROJECT_VERSION} -e LOCAL_AREA=${MRB_SOURCE} -e INSTALL_AREA=${MRB_INSTALL/\//} -e USER=${USER} \
           -e GRID=$GRID -e OFFSITE=$OFFSITE -e INPUT_FILE=${INPUT_FILE} -e INPUT_FILELIST=$(basename ${INPUT_FILELIST}) \
           -e IFDH_ART=${IFDH_ART} -e USESAMD=$USESAMD -e ROLE=$ROLE -e TIER=${DATA_TIER} -e SCHEMAS=$SCHEMAS -e CLEAN=$CLEAN \
           -e PROCESS_NAME=${PROCESS_NAME} \
           --memory=${MEMORYUSAGE} \
           --cpu=${CPU} \
           --disk=${DISKUSAGE} \
           ${LIFETIME_OPT} \
           ${TIMEOUT_OPT} \
           ${MAXCONCURRENT_OPT} \
           --resource-provides=usage_model=${USAGE_MODEL} \
           ${SITE_OPT} \
           ${SAMDATASET_OPT} \
           ${PROJECTNAME_OPT} \
           --lines='${LINES_OPT1}' \
           --lines='${LINES_OPT2}' \
           ${CVMFS_OPT} \
           ${NOVA_OPT} \
           --role=${ROLE} file://$PWD/submitGM2Data.sh ${INPUTLIST}"
   
   echo       | tee -a $LOG
   echo $CMD  | tee -a $LOG
   echo       | tee -a $LOG

   [ $TEST -eq 0 ] && $CMD 2>&1  | tee -a $LOG  && SUBMITTED=1
   
   if [ $SUBMITTED -eq 1 ]; then 
	echo -e "\nUse 'jobsub_history -G gm2 --user=$USER' to retrieve jobID history.\n" | tee -a $LOG 
   else
        echo -e "\n!!!Job submission failed! Please check above for errors!!!\n" | tee -a $LOG
   fi

else

   # set environment variable 
   RELEASE=${MRB_PROJECT_VERSION}
   TIER=${DATA_TIER}
   echo -e "\nsource ./submitGM2Data.sh ${INPUTLIST}..." | tee -a $LOG
   [ $TEST -eq 0 ] && source ./submitGM2Data.sh ${INPUTLIST}
fi

if [[ $USESAMD -eq 1 && $SUBMITTED -eq 1 ]]; then
   echo -e "-----------------------------------------------------------------------\n" | tee -a $LOG
   echo -e "SAM monitoring URL:\n" | tee -a $LOG
   echo -e "${SAM_MONITOR_URL}\n" | tee -a $LOG
   echo -e "SAM project URL:\n" | tee -a $LOG
   echo -e "${SAM_PROJECT_URL}\n" | tee -a $LOG
fi

[ $SUBMITTED -eq 1 ] && echo -e "\nSubmission done.\n" | tee -a $LOG

ifdh cp -D $LOG ${OUTPUT_DIR} 

if [ $? -ne 0 ]; then
   echo -e "\n!!!Unable to copy $(basename $LOG) to ${OUTPUT_DIR}!!!\n"
else
   ln -sf ${OUTPUT_DIR}/$(basename $LOG)
   [ $? -eq 0 ] && echo -e "\nCopied $(basename $LOG) to ${OUTPUT_DIR} and linked to it."
   rm -fr $TMPDIR
fi


echo -e "\nQuestions and problems? see $HELP for additional help."
echo -e "\nDone." 

