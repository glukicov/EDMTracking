# Wrapper script to submit a test job to the UK grid for testing (no input/output)
jobsub_submit -N 1 -M file:///gm2/app/users/glukicov/submitGridJobs/grid_submit.sh
# jobsub_submit -N 1 -G gm2 -M file:///gm2/app/users/glukicov/submitGridJobs/grid_submit.sh --expected-lifetime=1h --memory=500MB --disk=2GB --resource-provides=usage_model=OFFSITE --role=Analysis