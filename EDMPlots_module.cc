// //EDM tracker plots
// // Gleb 11 October 2019
// // Based of Geane plots module by Nick

// // Include needed ART headers
// #include "art/Framework/Core/EDAnalyzer.h"
// #include "art/Framework/Core/ModuleMacros.h"
// #include "art/Framework/Principal/Event.h"
// #include "art/Framework/Services/Optional/TFileService.h"
// #include "art/Framework/Services/Registry/ServiceHandle.h"

// //art records
// #include "gm2dataproducts/strawtracker/TrackArtRecord.hh"
// #include "gm2dataproducts/strawtracker/TrackDetailArtRecord.hh"
// #include "gm2dataproducts/mc/ghostdetectors/GhostDetectorArtRecord.hh"

// //Utils
// #include "gm2geom/common/Gm2Constants_service.hh"
// #include "gm2util/common/dataModuleDefs.hh"
// #include "gm2util/common/RootManager.hh"
// #include "gm2util/common/RunTypeTools.hh"
// #include "gm2util/coordSystems/CoordSystemUtils.hh"
// #include "gm2tracker/quality/TrackQuality_service.hh"
// #include "gm2geom/strawtracker/StrawTrackerGeometry_service.hh"
// #include "gm2tracker/utils/GeaneTrackUtils.hh"

// //C++ includes
// #include <iostream>
// #include <string>
// #include <vector>
// #include <map>
// #include <set>
// #include <math.h>

// namespace gm2analyses {

// class EDMPlots : public art::EDAnalyzer {

// public:

//   explicit EDMPlots(fhicl::ParameterSet const& pset);
//   void analyze(const art::Event& event ) override;
//   void beginJob() override;
//   void beginRun(art::Run const & r) override;
//   void endRun(art::Run const & r) override;
//   void endJob() override;

// private:

//   std::string name_;

//   //Producer labels
//   std::string TrackModuleLabel_;
//   std::string TrackInstanceName_;
//   std::string dcaDigitModuleLabel_;
//   std::string dcaDigitInstanceLabel_;
//   std::string DummyModuleLabel_;
//   std::string DummyInstanceName_;
//   std::string trajectoryModuleLabel_;
//   std::string trajectoryInstanceName_;

//   //ROOT plotting members
//   std::unique_ptr<RootManager> rootManager_;
//   std::string dirName_;

//   //Helper tools
//   gm2geom::CoordSystemsStoreData cs_;
//   gm2strawtracker::GeaneTrackUtils geaneTrackUtils_;
//   art::ServiceHandle<gm2strawtracker::TrackQuality> trackQuality_;

//   //Variables that we cut on
//   bool applyTrackQuality_;
//   double pValueCut_;
//   int numPlanesHitCut_;
//   double energyLossCut_;
//   vector<double> timeWindow_;
//   vector<double> momWindow_;
//   vector<int> stations_;

//   // Keep track of number of tracks and number of events
//   int tracksInRun_;
//   int eventsInRun_;

//   bool makeTruthPlots_; // decided in BoR 
//   bool useTrackDetailArtRecord_; // Flag for grabbing the detail track art record

//   // CS transforms in Bor 
//   gm2util::CoordSystemUtils csUtils_;
//   std::map< std::string, gm2geom::CoordSystemsStoreData > detCoordMap_;

//   // Data and MC plots (fitted)
//   void BookHistograms(TDirectory* dir);
//   void FillHistograms(const art::Event& event);

//   // MC-truth plots (from ghost detector)
//   void BookTruthHistograms(TDirectory* dir);
//   void FillTruthHistograms(const art::Event& event);


// }; //End of class EDMPlots

// EDMPlots::EDMPlots(fhicl::ParameterSet const& pset)
//   : art::EDAnalyzer(pset)
//   , name_( "EDMPlots" )
//   , TrackModuleLabel_( pset.get<std::string>("TrackModuleLabel", dataModuleDefs::trackModuleLabel()) )
//   , TrackInstanceName_( pset.get<std::string>("TrackInstanceName", dataModuleDefs::recoInstanceLabel()) )
//   , DummyModuleLabel_( pset.get<std::string>("DummyModuleLabel", dataModuleDefs::strawBuilderModuleLabel()) )
//   , DummyInstanceName_( pset.get<std::string>("DummyInstanceName", "trackerdummyplane") )
//   , rootManager_()
//   , dirName_( pset.get<std::string>("dirName", name_) )
//   , cs_()
//   , geaneTrackUtils_()
//   , applyTrackQuality_( pset.get<bool>("applyTrackQuality", true) )
//   , pValueCut_( pset.get<double>("pValueCut", 0.0) )
//   , numPlanesHitCut_( pset.get<int>("numPlanesHitCut", 0))
//   , energyLossCut_( pset.get<double>("energyLossCut", 1e6)) // default energy loss cut large
//   , timeWindow_(pset.get<vector<double> >("timeWindow", {}))
// , momWindow_(pset.get<vector<double> >("momWindow", {}))
// , stations_(pset.get<vector<int> >("stations", {}))
// , tracksInRun_(0)
// , eventsInRun_(0)
// , makeTruthPlots_();
// , useTrackDetailArtRecord_(pset.get<bool>("useTrackDetailArtRecord", false))
// , csUtils_()
// , detCoordMap_()
// {}

// void EDMPlots::beginJob() {
//   //Create a ROOT file and manager
//   art::ServiceHandle<art::TFileService> tfs;
//   auto& outputRootFile = tfs->file();
//   rootManager_.reset( new RootManager(name_, &outputRootFile) );

//   //Create directory structure (do it here so that they're in reasonable order)
//   auto topDir = rootManager_->GetDir(&outputRootFile, dirName_, true); //true -> create if doesn't exist
//   rootManager_->GetDir(topDir, "RunInfo", true);
//   rootManager_->GetDir(topDir, "Fit", true);
//   if (makeTruthPlots_) {
//     rootManager_->GetDir(topDir, "Truth", true);
//     rootManager_->GetDir(topDir, "dFitTruth", true);
//   }
//   // Book histograms
//   BookHistograms(topDir);
// }//beginJob


// void EDMPlots::beginRun(art::Run const & r) {

//   //make decision at BoR whether to make truth plots 
//   if(runTypeTools::dataRun(r)) makeTruthPlots_=false;
//   else makeTruthPlots_=true;

//   //Get coord systems
//   cs_ = artg4::dataFromRunOrService<gm2geom::CoordSystemsStoreData, gm2geom::CoordSystemsStore>
//         ( r, dataModuleDefs::coordSysModuleLabel(), dataModuleDefs::coordSysInstanceLabel() );
//   if ( cs_.size() == 0 ) mf::LogWarning(name_) << "This run does not contain any data associated with the coordinate system\n";

//   // Add extra coordinate system maps to speed up transforms
//   art::ServiceHandle<StrawTrackerGeometryService> sgeom_;
//   std::vector<std::string> detNames;
//   detNames.push_back("TrackerStation");
//   detNames.push_back("TrackerModule");
//   for (auto s : sgeom_->geom().whichScallopLocations) {
//     for (unsigned int m = 0; m < sgeom_->geom().getNumModulesPerStation(); ++m) {
//       detNames.push_back(Form("Module%d:%d", s, m));
//     }
//   }
//   csUtils_.setDetectorNames(detNames);
//   csUtils_.detNameToCoordMap(cs_, detCoordMap_);

//   // Reset tracks and events
//   eventsInRun_ = 0;
//   tracksInRun_ = 0;

// }//beginRun

// void EDMPlots::analyze(const art::Event& event) {

//   // Set flag for whether to fill truth plots
//   if (event.isRealData()) makeTruthPlots_ = false;

//   //Fill plots
//   FillHistograms(event);

//   // Increment event counter
//   eventsInRun_++;

// }//analyze


// void EDMPlots::endRun(art::Run const & r) {
//   auto runInfoDirectory = rootManager_->GetDir(rootManager_->GetDir(dirName_), "RunInfo");
//   auto tgTracks = rootManager_->Get<TGraph*>(runInfoDirectory, "TracksPerEvent" );
//   if (eventsInRun_ > 0) tgTracks->SetPoint(tgTracks->GetN(), r.run(), double(tracksInRun_) / eventsInRun_);
// }

// void EDMPlots::endJob() {

//   const char* boolYN[2] = {"No", "Yes"}; // True/False -> Yes/No mapping

//   //Dump some per job results
//   mf::LogInfo info("summary");
//   info << "\n--------- EDM Plots Report ---------\n";
//   info << "Total number of events = " << eventsInRun_ << "\n";
//   info << "Total number of track = " << tracksInRun_ << "\n";
//   info << "Track Quality Cuts applied: " << boolYN[applyTrackQuality_] << "\n";
//   info << "p-value cut used: " << pValueCut_ << "\n";
//   if (numPlanesHitCut_ > 0) info << "numPlanesHitCut_: " << numPlanesHitCut_ << "\n";
//   if ( energyLossCut_ < 1e6) info << "energyLossCut_: " << energyLossCut_ << "\n";
//   if ( timeWindow_.size() >= 2) info  <<  "timeWindow_: " << timeWindow_[0] << " " << timeWindow_[1] << "\n";
//   if ( momWindow_.size() >= 2)  info << "momWindow_: " << momWindow_[0] << " " << momWindow_[1] << "\n";
//   info << "------------------------------------------------\n";

//   //Clear out empty histos recursively from the top-level directory
//   rootManager_->ClearEmpty(rootManager_->GetDir(dirName_));

// } //endJob







// } // namespace gm2analyses

// DEFINE_ART_MODULE(gm2analyses::EDMPlots)
