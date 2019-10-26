#!/bin/env python

import os, sys, string, re, shutil, optparse


################################################################################
#
# set environment variables
#
################################################################################
project      = str(os.environ.get('MRB_PROJECT'))
release      = str(os.environ.get('MRB_PROJECT_VERSION'))
local_area   = str(os.environ.get('MRB_TOP')) 
install_area = str(os.environ.get('MRB_INSTALL'))
user         = str(os.environ.get('USER'))
qual         = str(os.environ.get('SETUP_ART')).split("-q ")[-1]

#####################################################
#
#
# Parser defines all the options
# for the DataParserUtils script.
#
#
#####################################################

class DataParserUtils( optparse.OptionParser ) :

   # constructor
   def __init__( self ) :
     formatter = optparse.TitledHelpFormatter( max_help_position=22, width = 190 )
     optparse.OptionParser.__init__(self, formatter = formatter ) #become an OptionParser

   # show the help menu
   def showHelp( self ):
     self.parse_args( "--help".split() )

   # options for data production
   def addProductionOpts( self ) :
     self.addCommonOpts()
     self.addDataTypeOpts()
     self.addProcessOpts()
     self.addDetectorOpts()
     self.addGunOpts()
     self.addParticleOpts()
     self.addHelpOpts()

   # data common arguments
   def addCommonOpts( self ) :
     common = optparse.OptionGroup( self, "Common Options" )
     common.add_option("--njobs", dest="njobs", type="int", default=-1, help="Number of jobs to process")
     common.add_option("--maxconcurrent", dest="maxconcurrent", type="int", default=-1, help="Max number of jobs running concurrently")
     common.add_option("--nevents", dest="nevents", type="int", default=-1, help="Number of events to process per job")
     common.add_option("--first-event",dest="first-event", type="int", default=1, help="First event to start the processing")
     common.add_option("--runnumber",dest="runNumber", type="int", default=1, help="Run number to start the processing with when --input-file/filelist is used")
     common.add_option("--campaign", dest="campaign", default="run", help="Campaign name [ commission, run1, <your own> ]")
     common.add_option("--sam-dataset", dest="sam-dataset", default="None", help="Name of the dataset for processing")
     common.add_option("--sam-max-files", dest="sam-max-files", type="int", default=-1, help="Number of files per job to process in the sam dataset")
     common.add_option("--nmuonsPerFill", dest="nmuonsPerFill", type="int", default=1, help="Number of muons per fill")
     common.add_option("--fhiclFile", dest="fhiclFile", default="None", help="Fhicl file(s) [ required, can also be comma,separated,list ]")
     common.add_option("--fcl", dest="fhiclFile", default="None", help="shorthand for --fhiclFile")
     common.add_option("--tag", dest="tag", default="None", help="Output filename tag, eg. gm2<tag>_<data_tier>.root")
     common.add_option("--noGrid", dest="noGrid", default=False, action="store_true", help="Run jobs on local machine (useful when testing or debugging).")
     common.add_option("--noifdh_art", dest="noifdh_art", default=False, action="store_true", help="Run jobs without using ifdh_art service")
     common.add_option("--localArea", dest="localArea", default=False, action="store_true", help="Use the local area "+local_area)
     common.add_option("--samelocal", dest="samelocal", default=False, action="store_true", help="Use the same local area from previous submission (no re-tarring)")
     common.add_option("--input-file", dest="input-file", default="None", help="Single input data file")
     common.add_option("--input-filelist", dest="input-filelist", default="None", help="File containing a list of input data files to read, one per line")
     common.add_option("--json", dest="json", default=False, action="store_true", help="Copy json files to the output directory")
     common.add_option("--output-dir", dest="output-dir", default="None", help="Output directory of the files")
     common.add_option("--onebyone", dest="onebyone", default=False, action="store_true", help="Produce one output file (set) for one input file (must use together with --noifdh_art).")
     common.add_option("--multipleroot", dest="multipleroot", default=False, action="store_true", help="Copy multiple root output files to the output directory (when multiple root outputs are produced).")
     common.add_option("--cleanup", dest="cleanup", default=False, action="store_true", help="Clean up everything after job completion.")
     common.add_option("--toplocaldir", dest="toplocaldir", default=False, action="store_true", help="Change local running directory to "+local_area+"/run")
     common.add_option("--release", dest="release", default=release, help="Release version to be used, default="+release)
     common.add_option("--qual", dest="qual", default=qual, help="Version qualifier to be used when running jobs locally, default="+qual)
     common.add_option("--offsite", dest="offsite", default="Yes", help="Submit jobs to open science grid in addition to FermiGrid, default=yes, set to no to submit to FermiGrid only")
     common.add_option("--offsite-only", dest="offsite-only", default=False, action="store_true", help="Submit jobs to the open science grid only")
     common.add_option("--site", dest="site", default="None", help="Submit jobs to these sites: comma,separated,list (site list: https://cdcvs.fnal.gov/redmine/projects/fife/wiki/Information_about_job_submission_to_OSG_sites)")
     common.add_option("--blacklist", dest="blacklist", default="None", help="DO NOT submit jobs to these sites: comma,separated,list (site list: https://cdcvs.fnal.gov/redmine/projects/fife/wiki/Information_about_job_submission_to_OSG_sites)")
     common.add_option("--memorytracker", dest="memorytracker", default=False, action="store_true", help="Turn on the art MemoryTracker service.")
     common.add_option("--debug", dest="debug", default=False, action="store_true", help="Debugging mode with --verbose enabled")
     common.add_option("--verbose", dest="verbose", default=False, action="store_true", help="Verbose printout for art jobs.")
     common.add_option("--trace", dest="trace", default=False, action="store_true", help="Activate tracing.")
     common.add_option("--timing",dest="timing", default=False, action="store_true", help="Activate monitoring of time spent per event/module.")
     common.add_option("--test", dest="test", default=False, action="store_true", help="Print out job submission commands only, no actual submission.")
     common.add_option("--memory", dest="memory", default=2, type="int", help="[Expert option] Request worker nodes have at least NUMBER of memory (NUMBER in GB, if NUMBER > 100, then in MB, default is 2).")
     common.add_option("--cpu", dest="cpu", default=1, type="int", help="[Expert option] Request worker nodes have at least NUMBER of CPUs (default is 1).")
     common.add_option("--core", dest="core", default="None", help="[Expert option] Run 'taskset -c' command on worker nodes to set processor affinity, can specify a single cpu or a list.")
     common.add_option("--disk", dest="disk", default=33, type="int", help="[Expert option] Request worker nodes have at least NUMBER of disk space (NUMBER in GB, default is 33).")
     common.add_option("--lifetime", dest="lifetime", default="8h", help="[Expert option] Expected lifetime of the job in NUMBER[UNITS] to match against resources (default is 8h).")
     common.add_option("--timeout", dest="timeout", default=-1, help="[Expert option] Kill job if still running after NUMBER[UNITS] of time (default is not to kill any job).")
     common.add_option("--requestid", dest="requestid", default="9999", help="[Expert option] Production request id as defined in the production google form or wiki.")
     common.add_option("--config", dest="config", default="nominal", help="[Expert option] Unique configuration for data production.")
     common.add_option("--process", dest="process", default="None", help="[Expert option] Change art program process name for the (last) fhiclFile (if a list is given).")
     common.add_option("--useProRole", dest="useProRole", default=False, action="store_true", help="[Expert option] Run as the Production Role (i.e. 'gm2pro').")
     common.add_option("--subgroup", dest="subgroup", default="None", help="[Expert option] Subgroup to use for job priorities and accounting.")
     common.add_option("--schemas", dest="schemas", default="xroot", help="[Expert option] Schemas to use for ifdh establishProcess, default is xroot, change to None to use raw cp")
     common.add_option("--schemas_uri", dest="schemas_uri", default="fndca1.fnal.gov/pnfs/fnal.gov/usr/", help="[Expert option] Schemas URI to use for ifdh establishProcess")
     common.add_option("--cvmfs", dest="cvmfs", default=True, action="store_true", help="[Expert option] GM2 CVMFS repository check, default True")
     common.add_option("--nova", dest="nova", default=False, action="store_true", help="[Expert option] Nova CVMFS repository check, default False")
     common.add_option("--sl6", dest="sl6", default=False, action="store_true", help="[Expert option] Use sigularity image for SL6 containers, default False")
     common.add_option("--lines", dest="lines", default="None", help="[Expert option] Add extra line(s) to Condor submission file")
     self.add_option_group( common )

   # data type arguments
   def addDataTypeOpts( self ):
     dtype = optparse.OptionGroup( self, "Data Type Options" )
     dtype.add_option("--mc", dest="mc", default=False, action="store_true", help="Run Simulated data")
     dtype.add_option("--daq", dest="daq", default=False, action="store_true", help="Run DAQ data")
     self.add_option_group( dtype )


   # processing type common arguments
   def addProcessOpts( self ) :
     process = optparse.OptionGroup( self, "Processing Type Options" )
     process.add_option("--unpack", dest="unpack", default=False, action="store_true", help="Run the unpacking")
     process.add_option("--truth", dest="truth", default=False, action="store_true", help="Run the g-2 ring simulation")
     process.add_option("--digit", dest="digit", default=False, action="store_true", help="Run the digitalization")
     process.add_option("--reco", dest="reco", default=False, action="store_true", help="Run the reconstruction")
     process.add_option("--full", dest="full", default=False, action="store_true", help="Run both the digitalization and reconstruction")
     process.add_option("--ana", dest="ana", default=False, action="store_true", help="Run the analyzer stage")
     self.add_option_group( process )

   # detector type
   def addDetectorOpts( self ) :
     detector = optparse.OptionGroup( self, "Detector Type Options" )
     detector.add_option("--ring", dest="ring", default=False, action="store_true", help="Run the g-2 ring simulation")
     detector.add_option("--harp", dest="harp", default=False, action="store_true", help="Run in Fiber Harps mode")
     detector.add_option("--calo", dest="calo", default=False, action="store_true", help="Run in Calorimeter mode")
     detector.add_option("--tracker", dest="tracker", default=False, action="store_true", help="Run in Tracker mode")
     detector.add_option("--kickeroff", dest="kickeroff", default=False, action="store_true", help="Run without Kicker")
     detector.add_option("--quadoff",   dest="quadoff",   default=False, action="store_true", help="Run without Quad")
     self.add_option_group( detector ) 

   # particle gun
   def addGunOpts( self ) :
     gun = optparse.OptionGroup( self, "Particle gun Type Options" )
     gun.add_option("--beamGun", dest="beamGun", default=False, action="store_true", help="Run the Beam gun")
     gun.add_option("--gasGun", dest="gasGun", default=False, action="store_true", help="Run the Gaussian Gas gun")
     gun.add_option("--miGun", dest="miGun", default=False, action="store_true", help="Run the Magic Inflector gun")
     gun.add_option("--simpleGun", dest="simpleGun", default=False, action="store_true", help="Run the Simple Source gun")
     self.add_option_group( gun )

   # particle type
   def addParticleOpts( self ) :
     particle = optparse.OptionGroup( self, "Particle gun Type Options" )
     particle.add_option("--muon", dest="muon", default=False, action="store_true", help="Run in muons mode")
     particle.add_option("--proton", dest="proton", default=False, action="store_true", help="Run in protons mode")
     particle.add_option("--pion", dest="pion", default=False, action="store_true", help="Run in pions mode")
     self.add_option_group( particle )

   # fill builder type

   # help options
   def addHelpOpts( self ) :
     menu = optparse.OptionGroup( self, "Help Menu" )
     menu.add_option("--help-common", dest="help_common", action="store_true", default=False, help="Help for common options")
     menu.add_option("--help-data", dest="help_data", action="store_true", default=False, help="Help for the type of data options [required]")
     menu.add_option("--help-process", dest="help_process", action="store_true", default=False, help="Help for processing options [required]")
     menu.add_option("--help-detector", dest="help_detector", action="store_true", default=False, help="Help for detector options")
     menu.add_option("--help-gun", dest="help_gun", action="store_true", default=False, help="Help for particle gun options")
     menu.add_option("--help-particle", dest="help_particle", action="store_true", default=False, help="Help for particle type options (muon is default)")
     self.add_option_group( menu )
   


#####################################################
#
#
# The main function
#
#
#####################################################
def main() :


    ######################################
    #
    #  help menu
    #
    ######################################   


    if sys.argv[1] == "--help" or sys.argv[1] == "-h" :
       parserHelp = DataParserUtils()
       parserHelp.addHelpOpts()
       parserHelp.showHelp()

    parser = DataParserUtils()
    parser.addProductionOpts()

    (opts, args) = parser.parse_args(sys.argv)
    opts = vars(opts)


    if opts["help_common"] :
       parserHelp = DataParserUtils()
       parserHelp.addCommonOpts()
       parserHelp.showHelp()
    
    if opts["help_data"] :
       parserHelp = DataParserUtils()
       parserHelp.addDataTypeOpts()
       parserHelp.showHelp()

    if opts["help_process"] :
       parserHelp = DataParserUtils()
       parserHelp.addProcessOpts()
       parserHelp.showHelp()

    if opts["help_detector"] :
       parserHelp = DataParserUtils()
       parserHelp.addDetectorOpts()
       parserHelp.showHelp()

    if opts["help_gun"] :
       parserHelp = DataParserUtils()
       parserHelp.addGunOpts()
       parserHelp.showHelp()

    if opts["help_particle"] :
       parserHelp = DataParserUtils()
       parserHelp.addParticleOpts()
       parserHelp.showHelp()


    # debug activated
    if opts["debug"]:
       opts["verbose"] = True
    
    # the code below is needed for the bash script
    if not opts["muon"] and not opts["proton"] and not opts["pion"] :
       opts["muon"] = True

    if not opts["ring"] and not opts["harp"] and not opts["calo"] and not opts["tracker"] :
       name = opts["fhiclFile"].lower()
    # only do this for single fcl mode
       if name.find(',') == -1:
          if name.find('gun') != -1 :
             opts["ring"] = True
          elif name.find('harp') != -1 :
             opts["harp"] = True
          elif name.find('calo') != -1 :
             opts["calo"] = True
          elif name.find('track') != -1 :
             opts["tracker"] = True
 

    output = ''
    for i in opts :
        key   = str(i)
        value = str(opts[i])
        output += key + ' '
        output += value + ' '

    print output


if __name__ == "__main__" :
   main()
