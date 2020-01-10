R__LOAD_LIBRARY(/gm2/app/users/scharity/Offline/Analysis/Blinding/dev_v9_30_00/build_slf6.x86_64/gm2util/lib/libgm2util_blinders.so)
#include "../Blinders.hh"

#include "TFile.h"
#include "TH1D.h"
#include "TF1.h"
#include "TTree.h"

blinding::Blinders::fitType ftype = blinding::Blinders::kOmega_a;

blinding::Blinders getBlinded( ftype, "please change this default blinding string" );

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

  TFile *file = new TFile("gm2offline_example.root");
  file->ls();

  TTree *clusterTree = (TTree*)file->Get("clusterTree/clusters");
  clusterTree->Draw("time*1.25/1000>>wiggle(2000,0,300)","energy>1800 && energy < 10000","goff");
  TH1D *wiggle = (TH1D*)gDirectory->Get("wiggle");
  
  TF1 *func = new TF1("func", blinded_wiggle, 30,280,5);
  
  func->SetParNames("N","#tau","A","R","#phi");
  
  func->SetParameters(1500,64.4,0.4,0,1);
  
  func->SetLineColor(2);
  
  func->SetNpx(1000);
  
  wiggle->Fit(func,"REM");

}
