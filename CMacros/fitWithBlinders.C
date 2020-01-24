// Author: Saskia
// Modified by Gleb (6 Jan 2020)
// Simple 5-parameter blinded fit
// Input: ROOT Tree data 

// Blinding libraries 
R__LOAD_LIBRARY(/Users/gleb/software/EDMTracking/Blinding/libBlinders.so)
#include "/Users/gleb/software/EDMTracking/Blinding/Blinders.hh"
using namespace blinding;

#include "TFile.h"
#include "TH1D.h"
#include "TF1.h"
#include "TTree.h"

Blinders::fitType ftype = Blinders::kOmega_a;

Blinders getBlinded( ftype, "EDM every day!" );

double blinded_wiggle(double *x, double *p){
  double norm = p[0];
  double life = p[1];
  double asym = p[2];
  double R = p[3];
  double phi = p[4];
  double time = x[0];
  double omega = getBlinded.paramToFreq(R);
  return norm * exp(-time/life) * (1 - asym*cos(omega*time + phi));
}

void fitWithBlinders(){

  TFile *file = new TFile("../DATA/Trees/60h_all_quality_tracks.root");
  file->ls();

  TTree *clusterTree = (TTree*)file->Get("QualityTracks");
  clusterTree->Draw("trackT0*1.25/1000>>wiggle(2000,0,300)","trackMomentum>1800 && trackMomentum < 10000","goff");
  TH1D *wiggle = (TH1D*)gDirectory->Get("wiggle");
  
  TF1 *func = new TF1("func", blinded_wiggle, 30,280,5);
  
  func->SetParNames("N","#tau","A","R","#phi");
  
  func->SetParameters(1500,64.4,0.4,0,1);
  
  func->SetLineColor(2);
  
  func->SetNpx(1000);
  
  wiggle->Fit(func,"REM");

}
