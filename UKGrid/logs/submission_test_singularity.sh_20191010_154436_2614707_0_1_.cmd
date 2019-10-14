universe          = vanilla
executable        = /fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229/submission_test_singularity.sh_20191010_154436_2614707_0_1_wrap.sh
arguments         = 
output                = /fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229/submission_test_singularity.sh_20191010_154436_2614707_0_1_cluster.$(Cluster).$(Process).out
error                 = /fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229/submission_test_singularity.sh_20191010_154436_2614707_0_1_cluster.$(Cluster).$(Process).err
log                   = /fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229/submission_test_singularity.sh_20191010_154436_2614707_0_1_.log
environment   = CLUSTER=$(Cluster);PROCESS=$(Process);CONDOR_TMP=/fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229;CONDOR_EXEC=/tmp;DAGMANJOBID=$(DAGManJobId);GRID_USER=gm2pro;IFDH_BASE_URI=http://samweb.fnal.gov:8480/sam/gm2/api;JOBSUBJOBID=$(CLUSTER).$(PROCESS)@jobsub03.fnal.gov;EXPERIMENT=gm2
rank                  = Mips / 2 + Memory
job_lease_duration = 3600
notification = always
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_output                 = True
transfer_output_files = .empty_file
transfer_error                  = True
transfer_executable         = True
transfer_input_files = /fife/local/scratch/uploads/gm2/gm2pro/2019-10-10_154435.038202_9229/submission_test_singularity.sh
+JobsubClientDN="/DC=org/DC=incommon/C=US/ST=IL/L=Batavia/O=Fermi Research Alliance/OU=Fermilab/CN=gm2pro-gm2gpvm01.fnal.gov"
+JobsubClientIpAddress="131.225.67.75"
+Owner="gm2pro"
+JobsubServerVersion="1.3.0.1.rc13"
+JobsubClientVersion="1.2.10"
+JobsubClientKerberosPrincipal="glukicov@FNAL.GOV"
+SingularityImage="/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl6:latest"
+JOB_EXPECTED_MAX_LIFETIME = 1800
notify_user = glukicov@fnal.gov
x509userproxy = /var/lib/jobsub/creds/proxies/gm2/x509cc_gm2pro_Production
+AccountingGroup = "group_gm2.gm2pro"
+Jobsub_Group="gm2"
+JobsubJobId="$(CLUSTER).$(PROCESS)@jobsub03.fnal.gov"
+Drain = False
+DESIRED_Sites = "Liverpool"
+GeneratedBy ="NO_UPS_VERSION jobsub03.fnal.gov"
+DESIRED_usage_model = "OFFSITE"
request_disk = 1GB
request_memory = 1000MB
requirements  = target.machine =!= MachineAttrMachine1 && target.machine =!= MachineAttrMachine2 && ((isUndefined(target.GLIDEIN_Site) == FALSE) && (stringListIMember(target.GLIDEIN_Site,my.DESIRED_Sites))) && (isUndefined(DesiredOS) || stringListsIntersect(toUpper(DesiredOS),IFOS_installed)) && (stringListsIntersect(toUpper(target.HAS_usage_model), toUpper(my.DESIRED_usage_model)))


queue 1