// Author: Saskia
// Modified by Gleb (6 Jan 2020)
// Test the "injection" EDM blinding
// run with e.g. "root edmBlindingTest.C"

// Blinding libraries 
R__LOAD_LIBRARY(/Users/gleb/software/EDMTracking/Blinding/libBlinders.so)
#include "/Users/gleb/software/EDMTracking/Blinding/Blinders.hh"
using namespace blinding;

//write array to file
#include <fstream>
#include <iterator>
#include <vector>
std::vector<long double> sanity;

//ROOT includes
#include "TF1.h"
#include "TTree.h" // always needed for MD5 lib see see https://cdcvs.fnal.gov/redmine/projects/gm2analyses/wiki/Library_installation
#include "TFile.h"

//Input constants
int NInject = 10000;

// edm constants
long double e = 1.6e-19; // J
long double aMu = 11659208.9e-10;
long double mMu = 105.6583715; // u
long double mMuKg = mMu * 1.79e-30; // kg
long double B = 1.451269; // T
long double c = 299792458.; // m/s
long double cm2m = 100.0;
long double hbar = 1.05457e-34;
long double pmagic = mMu / std::sqrt(aMu);
long double gmagic = std::sqrt( 1. + 1. / aMu );
long double beta   = std::sqrt( 1. - 1. / (gmagic*gmagic) );

// defaults for now
long double d0 = 1.9e-19; // BNL edm limit in e.cm
long double ppm = 1e-6;

// blinding constructor constants
long double R = 3.5; // * d0
long double boxWidth = 0.3;
long double gausWidth = 0.8;
Blinders::fitType ftype = Blinders::kOmega_a;

long double blinded_edm() {

  // std::cout << "R = " << R << "\n";
  // std::cout << "boxWidth = " << boxWidth << "\n";
  // std::cout << "gausWidth = " << gausWidth << "\n";

  TCanvas* c1 = new TCanvas();
  TH1F* h1 = new TH1F("CEDMBlind", "", 100., -1, 10);

  for (int i(0); i < NInject; i++) {
    std::string blindString = Form("There we go %d", i);
    // std::string blindString = "sameSame";
    // std::cout << "\nblindString: " << blindString << "\n";
    Blinders iBlinder( ftype, blindString , boxWidth, gausWidth);

    // use the blinder to give us a blinded offset from the input R which is 10*d0
    long double iAmp = iBlinder.paramToFreq(R); // this is something like 1.442 which is the shifted / blinded omegaA value
    long double iRef  = iBlinder.referenceValue(); // this is 1.443 which is the reference omegaA value
    long double iDiff =  ((iAmp / iRef) - 1) / ppm; // this is (R - Rref) in units of ppm

    // std::cout << "iDiff = " << iDiff << "\n";
    // std::cout << "iRef = " << iRef << "\n";
    // std::cout << "iAmp = " << iAmp << "\n";

    // iDiff tells us the edm value we have got
    // e.g. input EDM_blind = iDiff * d0
    long double EDM_blind = iDiff * d0;
    std::cout << "EDM_blind = " << EDM_blind << "\n";

    long double omegaA = iRef; // just use the reference value

    long double eta_blind = ((4 * mMuKg * c * EDM_blind) / (hbar * cm2m) );
    long double tan_delta_blind = (eta_blind * beta) / (2 * aMu);
    long double delta_blind = atan(tan_delta_blind);

    // std::cout << "aMu = " << aMu << "\n";
    // std::cout << "beta = " << beta << "\n";
    // std::cout << "delta_blind = " << delta_blind << "\n";
    // std::cout << "eta_blind = " << eta_blind << "\n";
    // std::cout << "tan_delta_blind = " << tan_delta_blind << "\n";

    h1->Fill(iDiff);
    sanity.push_back(iDiff);
    std::cout << "iDiff" << iDiff << "\n";

  }

  TFile* file = new TFile("../DATA/misc/CEDMBlind.root", "new");

  h1->GetXaxis()->SetTitle("R_{blind} - R_{ref} (ppm)");
  h1->GetXaxis()->CenterTitle();
  Int_t bin = h1->GetXaxis()->GetNbins();
  std::cout << "bins: " << bin << "\n";
  h1->Draw();
  h1->Write();
  c1->SaveAs("../fig/blindTest.png");

  file->Close();

  //write to file using a vector iterator
  std::ofstream output_file("../DATA/misc/CEDMBlind.txt");
  output_file << std::setprecision(15);
  std::ostream_iterator<long double> output_iterator(output_file, "\n");
  std::copy(sanity.begin(), sanity.end(), output_iterator);


  return 0.0;//norm * exp(-time/life) * (1 - asym*cos(omega*time + phi));
}

void edmBlindingTest() {
  double test = blinded_edm();
}
