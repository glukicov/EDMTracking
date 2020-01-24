// Author: Saskia
// Modified by Gleb (6 Jan 2020)

// Blinding libraries 
R__LOAD_LIBRARY(/Users/gleb/software/EDMTracking/Blinding/libBlinders.so)
#include "/Users/gleb/software/EDMTracking/Blinding/Blinders.hh"
using namespace blinding;

#include "TFile.h"
#include "TH1F.h"
#include "TF1.h"
#include "TTree.h"
#include "TVirtualFFT.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TGraph.h"
#include "TGraphErrors.h"

#include <sstream>

TH1* FFTHisto(TH1F* hist);
TH1D* SetupFFT(TH1* h, double xmin, double xmax);
TH1D* RescaleAxis(TH1* input, Double_t Scale);
double getDelta(double dMu);
double getDmu(double delta);

// edm constants
double e = 1.6e-19; // J
double aMu = 11659208.9e-10; 
double mMu = 105.6583715; // u
double mMuKg = mMu * 1.79e-30; // kg
double B = 1.451269; // T
double c = 299792458.; // m/s
double cm2m = 100.0;
double hbar = 1.05457e-34;
double pmagic = mMu/std::sqrt(aMu);
double gmagic = std::sqrt( 1.+1./aMu );
double beta   = std::sqrt( 1.-1./(gmagic*gmagic) );
double d0 = 1.9e-19; // BNL edm limit in e.cm
double ppm = 1e-6;
double asymFactor = 0.1;

// edm to use for testing things, know things about this amplitude as we simulated it
double TESTEDM = d0 / 2.; 

// for plotting
std::string titleStr = "";
std::string title = "";
std::stringstream oss;
std::string plotname = "";

// blinding parameters
// blind by having a width of 2*d0 around R=10*d0 with broad gaussian tails
double R = 3.5; 
double boxWidth = 0.25;
double gausWidth = 0.7;
Blinders::fitType ftype = Blinders::kOmega_a;
Blinders getBlinded(ftype, "please change this default blinding string", boxWidth, gausWidth);
// make a blinder for testing the unblinding procedure with a different plot
Blinders getBlindedTest(ftype,"testing blinding with py vs p plot", boxWidth, gausWidth);
//Blinders getBlindedTest(ftype,"do not use this string", boxWidth, gausWidth);

double blinded_edm_value( bool test = false ) {
  
  //
  // returns an blinded input edm value. returned dMu will be unphysical. it will be in the range of +- 3*d0 centred around 10*d0
  //

  double omega_blind = getBlinded.paramToFreq(R); // this is the blinded omegaA value
  double omega_ref   = getBlinded.referenceValue(); // this is the reference omegaA value
  double omega_diff  =  ((omega_blind / omega_ref) - 1) / ppm; // this is (omega_blind - omega_ref) in units of ppm
  double dMu_blind   = omega_diff * d0; // this is the blinded dMu in e.cm
  
  if (!test)  {
    return dMu_blind;
  }
  else {
    return TESTEDM;
  }

}

double blinded_test_osc( ) {

  // this returns a blinded ppm value
  
  double omega_blind = getBlindedTest.paramToFreq(R); // this is the blinded omegaA value
  double omega_ref   = getBlindedTest.referenceValue(); // this is the reference omegaA value
  double omega_diff  =  ((omega_blind / omega_ref) - 1) / ppm; // this is (omega_blind - omega_ref) in units of ppm
  double amp_blind   = omega_diff * 0.0001; // this is the blinded amplitude of the test oscillation
  return amp_blind;

}

double osc_func( double *x, double *p )  {
  double off = p[3];
  
  double amp  = p[0];
  double freq = p[1];
  double phi  = p[2];
  double time = x[0] + off;
  return (-amp * cos(freq * time + phi));
}

double omega_func( double *x, double *p )  {
  double off = p[5];
  
  double y0 = p[0];
  double tau = p[1];
  double amp  = p[2];
  double freq = p[3];
  double phi  = p[4];
  double time = x[0] - off;
  return y0 * exp(-time/tau) * ( 1 - amp * cos(freq * time + phi));
}

void TestOscBlindingWithOmegaA(){

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // make a canvas
  TCanvas* c1 = new TCanvas();

  // get the data
  TFile* file = new TFile();
  TDirectoryFile* cuts;
  
  TFile* f1 = new TFile("../gm2TrackerEDMPlots_60hr.root");
  titleStr = "60hr dataset";
  plotname = "h_thetay_vs_time_reco";
  file = f1;
  TDirectoryFile* edm = (TDirectoryFile*)file->Get("EDM");
  TDirectoryFile* vol = (TDirectoryFile*)edm->Get("noVolEvents");
  cuts = (TDirectoryFile*)vol->Get("0<p<3100MeV");
  cuts->ls();

  // get the plot
  TH2F* h_pyVsT = (TH2F*)cuts->Get(plotname.c_str());
  
  //
  // make an N(t) plot to get the omega a phase
  //

  TProfile* p_pyVsT = (TProfile*)h_pyVsT->ProfileX();
  int nbinsx = p_pyVsT->GetNbinsX();
  double binw = p_pyVsT->GetBinWidth(1);
  int x1 = p_pyVsT->GetXaxis()->GetBinLowEdge(0);
  int x2 = p_pyVsT->GetXaxis()->GetBinUpEdge(nbinsx-1);
  TH1F* h_nt = new TH1F("h_nt","N(t) vs t",nbinsx,x1,x2);

  // need to get typical sigma from this plot too
  double sigmas[nbinsx];
  int nonZero = 0;
  for (int ibin(0); ibin<p_pyVsT->GetNbinsX(); ibin++) {
    double ients = p_pyVsT->GetBinEntries(ibin);
    h_nt->SetBinContent(ibin,ients);
    if (nonZero == 0 && h_nt->GetBinContent(ibin) > 0) {
      nonZero = ibin;    
    }      
    sigmas[ibin-nonZero] = p_pyVsT->GetBinError(ibin) * sqrt(p_pyVsT->GetBinEntries(ibin));
  }
  
  float xmin = 40.0;//p_pyVsT->GetXaxis()->GetXmin();
  float xmax = 60.0;//p_pyVsT->GetXaxis()->GetXmax();

  h_nt->GetXaxis()->SetRangeUser(xmin,xmax);

  TF1* ftmp = new TF1("ftmp",omega_func,xmin,xmax,6);
  ftmp->SetParNames("N","#gamma#tau","A","#omega","#phi","offset");
  ftmp->SetParameter(0,200e3);
  ftmp->SetParLimits(0,190e3,220e3);
  ftmp->SetParameter(1,64.4);
  ftmp->SetParLimits(2,63,65);
  ftmp->SetParLimits(2,0.0,0.3);
  ftmp->SetParameter(3,getBlinded.referenceValue());
  ftmp->FixParameter(3,getBlinded.referenceValue());
  ftmp->SetParLimits(4,0,2*M_PI);
  ftmp->SetParameter(5,xmin);
  ftmp->FixParameter(5,xmin);
  ftmp->SetLineColor(2);
  ftmp->SetNpx(10000);
  h_nt->Fit(ftmp);
  h_nt->Draw();
  c1->SaveAs("plots/h_ntVsT.eps");
  
  //
  // add an oscillation with the blinded amplitude 
  //   

  // make a new histogram with the required oscillation
  double freq = getBlinded.referenceValue(); // omega
  double phi_a = ftmp->GetParameter(4); // get the phase from the n(t) fit
  double phase = phi_a + M_PI/2.;
  double off = xmin;

  //
  // for testing, make a fake EDM oscillation with known amplitude, and add in the VBO so we get something similar
  //
  
  double fakeDelta = getDelta(TESTEDM);
  double fakeTanAmp = tan(fakeDelta) / gmagic;
  double fakeAmp = asymFactor * atan(fakeTanAmp); 
  double fakePhase = phase; // same as blinded EDM
  double fakeOff = xmin;

  TF1* fakeOsc = new TF1("fakeOsc",osc_func,xmin,xmax,4);
  fakeOsc->SetParNames("A","#omega_{a BNL}","#phi","offset");
  fakeOsc->SetParameters(fakeAmp,freq,fakePhase,fakeOff);
  fakeOsc->SetLineColor(2);
  fakeOsc->SetNpx(10000);
  fakeOsc->SetTitle("Toy MC, no blinding");
  fakeOsc->Draw();
  c1->SaveAs("testplots/fakeOsc.eps");

  double avbo = 0.0002;
  double fvbo = 2.208;// MHz
  double phi_vbo = phi_a * 0.3;

  double omega_vbo = 2 * M_PI * fvbo;
  TF1* vboOsc = new TF1("vboOsc",osc_func,xmin,xmax,4);
  vboOsc->SetParNames("A_{vbo}","f_{vbo}","#phi","offset");
  vboOsc->SetParameters(avbo,omega_vbo,phi_vbo,xmin); // f in MHz
  vboOsc->SetNpx(10000);
  vboOsc->Draw();
  fakeOsc->Draw("SAME");
  c1->SaveAs("testplots/vboOsc.eps");

  int nentries = 1e6;
  int npoints  = 1000;
  std::cout << "npoints = " << npoints << "\n";

  double xvals[npoints];
  double yvals[npoints];
  double xerrs[npoints];
  double yerrs[npoints];

  // draw the unblinded plot
  TF1* preBlindingOsc = new TF1("preBlindingOsc",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5]) ",xmin,xmin),xmin,xmax);
  preBlindingOsc->SetParameters(fakeAmp,freq,fakePhase,avbo,omega_vbo,phi_vbo); // f in MHz
  preBlindingOsc->SetNpx(10000);

  xvals[0] = xmin;
  for (int i(0); i<npoints; i++){
    if (i > 0) xvals[i] = xvals[i-1] + (xmax - xmin)/npoints;
    yvals[i] = preBlindingOsc->Eval(xvals[i]);
    double binE = sigmas[i] / sqrt(nentries);
    xerrs[i] = 0.0;
    yerrs[i] = binE;
  }
  preBlindingOsc->SetLineColor(kOrange+7);

  TGraphErrors* preBlinding = new TGraphErrors(npoints,xvals,yvals,xerrs,yerrs);
  preBlinding->SetMarkerStyle(8);
  preBlinding->SetMarkerSize(0.2);
  preBlinding->SetMarkerColor(kBlue+1);
  preBlinding->SetLineColor(kBlue+1);
  preBlinding->GetXaxis()->SetRangeUser(xmin,xmax);
  preBlinding->GetYaxis()->SetRangeUser(-4e-4,4e-4);
  preBlinding->SetTitle("VBO + known EDM (pre-blinding)");
  preBlinding->GetXaxis()->SetTitle("Time [#mus]");
  preBlinding->GetYaxis()->SetTitle("#theta_{y} [rad]");
  preBlinding->Fit(preBlindingOsc);
  preBlinding->Draw("AP");
  c1->SaveAs("testplots/preBlinding.eps");

  // add a blinded EDM oscillation
  double amp_blind = blinded_test_osc();
  double freq_blind = freq;
  double phase_blind = phi_a + M_PI/2;

  TF1* blindedOsc = new TF1("blindedOsc",osc_func,xmin,xmax,4);
  blindedOsc->SetParNames("A_{blind}","f_{blind}","#phi","offset");
  blindedOsc->SetParameters(amp_blind,freq_blind,phase_blind,xmin);
  
  TF1* totOsc = new TF1("totOsc",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5] ) - [6] * cos( [7] * (x+%.1f) + [8] )",xmin,xmin,xmin),xmin,xmax);
  totOsc->SetParameters(fakeAmp,freq,fakePhase,avbo,omega_vbo,phi_vbo,amp_blind,freq_blind,phase_blind); // f in MHz
  totOsc->SetNpx(10000);
  //  totOsc->GetXaxis()->SetRangeUser(xmin,xmax);
  totOsc->Draw();
  c1->SaveAs("testplots/totOsc.eps");
 
  xvals[0] = xmin;
  for (int i(0); i<npoints; i++){
    if (i > 0) xvals[i] = xvals[i-1] + (xmax - xmin)/npoints;
    yvals[i] = totOsc->Eval(xvals[i]);
    double binE = sigmas[i] / sqrt(nentries);
    xerrs[i] = 0.0;
    yerrs[i] = binE;
  }
  TGraphErrors* testGraph = new TGraphErrors(npoints,xvals,yvals,xerrs,yerrs);
  testGraph->SetMarkerStyle(8);
  testGraph->SetMarkerSize(0.2);
  testGraph->SetMarkerColor(kBlue+1);
  testGraph->SetLineColor(kBlue+1);
  testGraph->GetXaxis()->SetRangeUser(xmin,xmax);
  testGraph->SetTitle("VBO + known EDM + blinded EDM");
  testGraph->GetXaxis()->SetTitle("Time [#mus]");
  testGraph->GetYaxis()->SetTitle("#theta_{y} [rad]");
  testGraph->Draw("AP");
  c1->SaveAs("testplots/testGraph.eps");

  // turn this into a histogram to FFT
  TH1F* hist_fft = new TH1F("hist_fft","",npoints,xvals[0],xvals[npoints-1]);
  for (int i(0);i<npoints;i++) {
    hist_fft->SetBinContent(i,testGraph->Eval(xvals[i]));
    //hist_fft->SetBinError(i,testGraph->GetErrorY(i));
    hist_fft->SetBinError(i,yerrs[i]);
  }
  hist_fft->SetLineWidth(2);
  hist_fft->SetLineColor(kRed+1);
  hist_fft->SetMarkerColor(kBlue+1);
  hist_fft->SetMarkerStyle(8);
  hist_fft->SetMarkerSize(0.4);
  hist_fft->SetTitle("VBO + known EDM + blinded EDM");
  hist_fft->GetXaxis()->SetTitle("Time [#mus]");
  hist_fft->GetYaxis()->SetTitle("#theta_{y} [rad]");
  hist_fft->Draw("e1p");
  c1->SaveAs("testplots/hist_fft.eps");
  
  // FFT the plot to see the frequencies
  TH1 *h_FFT = 0;
  TVirtualFFT::SetTransform(0);
  TH1D* hist_fft_setup = SetupFFT(hist_fft, xmin, xmax);
  hist_fft_setup->Draw("HIST");
  c1->SaveAs("testplots/hist_fft_setup.eps");

  h_FFT = hist_fft_setup->FFT(h_FFT,"MAG");
  TH1D* hist_FFT_rescale = RescaleAxis(h_FFT,1./(xmax-xmin));
  hist_FFT_rescale->SetTitle("FFT of known EDM + blinded EDM + VBO");
  hist_FFT_rescale->GetXaxis()->SetTitle("Frequency [MHz]");
  nbinsx = hist_FFT_rescale->GetXaxis()->GetNbins();
  //hist_FFT_rescale->GetXaxis()->SetRangeUser(0,hist_FFT_rescale->GetXaxis()->GetBinUpEdge(nbinsx-1)/2);
  hist_FFT_rescale->GetXaxis()->SetRangeUser(0,4);
  hist_FFT_rescale->Draw("HIST");
  c1->SaveAs("testplots/hist_FFT_result.eps");
  
  // fit the graph
  TF1* fit_graph = new TF1( "fit_graph",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5] ) ",xmin,xmin),xmin,xmax);
  
  // fix all params except amplitudes --> know frequency from FFT

  // first one is the EDM osc which is now large with the blinding
  //  fit_graph->SetParameter(0,4.e-4);
  //  fit_graph->SetParLimits(0,1.e-4,6.e-4);

  fit_graph->SetParameter(1,freq);
  fit_graph->FixParameter(1,freq);

  fit_graph->SetParameter(2,fakePhase);
  fit_graph->FixParameter(2,fakePhase);

  // this is the vbo -- fast osc
  //fit_graph->SetParameter(3,0.0002);
  //fit_graph->SetParLimits(3,0.0001,0.0004);

  fit_graph->SetParameter(4,omega_vbo);
  fit_graph->SetParLimits(4,omega_vbo*0.8,omega_vbo*1.2);

  //fit_graph->SetParameter(5,phi_vbo);
  //fit_graph->FixParameter(5,phi_vbo);
  
  TGraphErrors* toFit = (TGraphErrors*)testGraph->Clone();

  toFit->SetMarkerStyle(8);
  toFit->SetMarkerSize(0.4);
  toFit->SetMarkerColor(kBlue+1);

  toFit->SetLineColor(kBlue+1);
  toFit->SetLineWidth(1);
  //  toFit->GetXaxis()->SetTitle("TWAT");

  fit_graph->SetLineColor(kRed);
  fit_graph->SetNpx(10000);

  toFit->Fit(fit_graph,"R");
  toFit->Draw("AP");

  toFit->GetXaxis()->SetRangeUser(xmin,xmax);
  c1->SaveAs("testplots/testGraphFit.eps");

  // the total edm amplitude (input + blinded) is equal to the fitted amplitude from that fit

  // subtract the blinding from the plot and refit
  double y_unb[npoints];
  double yerr_unb[npoints];
  for (int ient(0); ient<npoints; ient++){
    y_unb[ient] = testGraph->Eval(xvals[ient]) - blindedOsc->Eval(xvals[ient]); 
    yerr_unb[ient] = testGraph->GetErrorY(ient);
  }

  double totalEDM = fit_graph->GetParameter(0);
  double unblindedEDM = totalEDM - amp_blind;

  TF1* unblindedOsc = new TF1( "unblindedOsc",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5] ) ",xmin,xmin),xmin,xmax);
  
  // let the EDM amplitude float
  //unblindedOsc->SetParameter(0,2e-6);
  //unblindedOsc->SetParLimits(0,9e-7,6e-6);
  unblindedOsc->SetParameter(1,freq);
  unblindedOsc->FixParameter(1,freq);

  unblindedOsc->SetParameter(2,fakePhase);
  unblindedOsc->FixParameter(2,fakePhase);
  //unblindedOsc->SetParameter(3,fit_graph->GetParameter(3));
  //unblindedOsc->FixParameter(3,fit_graph->GetParameter(3));

  unblindedOsc->SetParameter(4,omega_vbo);
  unblindedOsc->SetParLimits(4,omega_vbo*0.8,omega_vbo*1.2);
  //unblindedOsc->SetParameter(5,phi_vbo);
  //unblindedOsc->FixParameter(5,phi_vbo);

  unblindedOsc->SetLineColor(kOrange+7);
  unblindedOsc->SetNpx(10000);

  c1->Clear();
  TGraphErrors* unblindedGr = new TGraphErrors(npoints,xvals,y_unb,xerrs,yerr_unb);
  unblindedGr->Fit(unblindedOsc,"R");
  unblindedGr->SetTitle("VBO + known EDM (after unblinding)");
  unblindedGr->GetXaxis()->SetTitle("Time [#mus]");
  unblindedGr->GetYaxis()->SetTitle("#theta_{y} [rad]");
  unblindedGr->Draw("AP");
  unblindedGr->GetXaxis()->SetRangeUser(xmin,xmax);
  unblindedGr->GetYaxis()->SetRangeUser(-4e-4,4e-4);
  unblindedGr->SetMarkerStyle(8);
  unblindedGr->SetMarkerSize(0.4);
  unblindedGr->SetMarkerColor(kBlue+1);
  unblindedGr->SetLineColor(kBlue+1);
  unblindedGr->SetLineWidth(1.0);
  c1->SaveAs("testplots/graphUnblinded.eps");

  // calculate some stuff
  double finalAmp = unblindedOsc->GetParameter(0);
  finalAmp = (finalAmp / asymFactor) * gmagic;
  double finalDmu = getDmu(finalAmp);
  std::cout << "finalAmp = " << unblindedOsc->GetParameter(0) << " finalDmu = " << finalDmu << " inputDmu = " << TESTEDM << "\n";

  //
  // add something in phase with g-2 at omega_a freq
  //

  TF1* totOscWithOmegaA = new TF1("omegaAosc",Form("-[0] * cos( [1] * (x+%.1f) + [2] )",xmin),xmin,xmax);
  double newAmp = 3e-4;
  totOscWithOmegaA->SetParameters(newAmp,freq,phi_a); // f in MHz
  totOscWithOmegaA->SetNpx(npoints);
  totOscWithOmegaA->Draw();
  c1->SaveAs("testplots/omegaAosc.eps");

  TF1* test1 = (TF1*)totOsc->Clone("aa"); 
  TF1* test2 = (TF1*)totOscWithOmegaA->Clone("bb");
  TF1* test3 = new TF1("cc","aa+bb",xmin,xmax);
  test3->SetNpx(npoints);
  test3->SetLineColor(kMagenta);
  test3->Draw();
  c1->SaveAs("testplots/test3.eps");

  TGraph* testGraphWithOmegaA_0 = new TGraph(test3);//npoints,xvals,yvals,xerrs,yerrs);
  TGraphErrors* testGraphWithOmegaA = new TGraphErrors(npoints,testGraphWithOmegaA_0->GetX(),testGraphWithOmegaA_0->GetY(),xerrs,yerrs);//npoints,xvals,yvals,xerrs,yerrs);
  testGraphWithOmegaA->SetMarkerStyle(8);
  testGraphWithOmegaA->SetMarkerSize(0.2);
  testGraphWithOmegaA->SetMarkerColor(kBlue+1);
  testGraphWithOmegaA->SetLineColor(kBlue+1);
  testGraphWithOmegaA->GetXaxis()->SetRangeUser(xmin,xmax);
  testGraphWithOmegaA->SetTitle("VBO + known EDM + blinded EDM + g-2");
  testGraphWithOmegaA->GetXaxis()->SetTitle("Time [#mus]");
  testGraphWithOmegaA->GetYaxis()->SetTitle("#theta_{y} [rad]");
  testGraphWithOmegaA->Draw("AP");
  test3->Draw("SAME");
  c1->SaveAs("testplots/testGraphWithOmegaA.eps");

  TF1* test4 = (TF1*)preBlindingOsc->Clone("dd"); 
  TF1* test5 = (TF1*)totOscWithOmegaA->Clone("ee");
  TF1* test6 = new TF1("ff","dd+ee",xmin,xmax);
  test6->SetNpx(npoints);
  test6->SetLineColor(kMagenta);
  test6->Draw();
  c1->SaveAs("testplots/test6.eps");

  //for (int i(0); i<npoints; i++){
  //  yvals[i] = preBlindingOsc->Eval(xvals[i]) + totOscWithOmegaA->Eval(xvals[i]);
  //  double binE = sigmas[i] / sqrt(nentries);
  //  xerrs[i] = 0.0;
  //  yerrs[i] = binE;
  //}

  TGraph* preBlindingWithOmegaA_0 = new TGraph(test6);
  TGraphErrors* preBlindingWithOmegaA = new TGraphErrors(npoints,preBlindingWithOmegaA_0->GetX(),preBlindingWithOmegaA_0->GetY(),xerrs,yerrs);
  preBlindingWithOmegaA->SetMarkerStyle(8);
  preBlindingWithOmegaA->SetMarkerSize(0.2);
  preBlindingWithOmegaA->SetMarkerColor(kBlue+1);
  preBlindingWithOmegaA->SetLineColor(kBlue+1);
  preBlindingWithOmegaA->GetXaxis()->SetRangeUser(xmin,xmax);
  preBlindingWithOmegaA->SetTitle("VBO + known EDM + g-2");
  preBlindingWithOmegaA->GetXaxis()->SetTitle("Time [#mus]");
  preBlindingWithOmegaA->GetYaxis()->SetTitle("#theta_{y} [rad]");
  preBlindingWithOmegaA->Draw("AP");
  c1->SaveAs("testplots/preBlindingWithOmegaA.eps");

  // turn the blinded graph into a histogram to FFT
  TH1F* hist_omegaA_fft = new TH1F("hist_omegaA_fft","",npoints,xvals[0],xvals[npoints-1]);
  for (int i(0);i<npoints;i++) {
    hist_omegaA_fft->SetBinContent(i,testGraphWithOmegaA->Eval(xvals[i]));
    //hist_omegaA_fft->SetBinError(i,testGraph->GetErrorY(i));
    hist_omegaA_fft->SetBinError(i,yerrs[i]);
  }
  hist_omegaA_fft->SetLineWidth(2);
  hist_omegaA_fft->SetLineColor(kRed+1);
  hist_omegaA_fft->SetMarkerColor(kBlue+1);
  hist_omegaA_fft->SetMarkerStyle(8);
  hist_omegaA_fft->SetMarkerSize(0.4);
  hist_omegaA_fft->SetTitle("VBO + known EDM + blinded EDM");
  hist_omegaA_fft->GetXaxis()->SetTitle("Time [#mus]");
  hist_omegaA_fft->GetYaxis()->SetTitle("#theta_{y} [rad]");
  hist_omegaA_fft->Draw("e1p");
  c1->SaveAs("testplots/hist_omegaA_fft.eps");
  
  // FFT the plot to see the frequencies
  TH1 *h_omegaA_FFT = 0;
  TVirtualFFT::SetTransform(0);
  TH1D* hist_omegaA_fft_setup = SetupFFT(hist_omegaA_fft, xmin, xmax);
  hist_omegaA_fft_setup->Draw("HIST");
  c1->SaveAs("testplots/hist_omegaA_fft_setup.eps");

  h_omegaA_FFT = hist_omegaA_fft_setup->FFT(h_omegaA_FFT,"MAG");
  TH1D* hist_omegaA_fft_rescale = RescaleAxis(h_omegaA_FFT,1./(xmax-xmin));
  hist_omegaA_fft_rescale->SetTitle("FFT of known EDM + blinded EDM + VBO");
  hist_omegaA_fft_rescale->GetXaxis()->SetTitle("Frequency [MHz]");
  nbinsx = hist_omegaA_fft_rescale->GetXaxis()->GetNbins();
  //hist_omegaA_fft_rescale->GetXaxis()->SetRangeUser(0,hist_omegaA_fft_rescale->GetXaxis()->GetBinUpEdge(nbinsx-1)/2);
  hist_omegaA_fft_rescale->GetXaxis()->SetRangeUser(0,4);
  hist_omegaA_fft_rescale->Draw("HIST");
  c1->SaveAs("testplots/hist_omegaA_fft_result.eps");

  // Fit it using FFT frequencies as guesses
  //  totOsc->SetParameters(fakeAmp,freq,fakePhase,avbo,omega_vbo,phi_vbo,amp_blind,freq_blind,phase_blind); // f in MHz
  TF1* fit_graph_withOmegaA = new TF1( "fit_graph_withOmegaA",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5] - [6] * cos( [7] * (x+%.1f) + [8] ) )",xmin,xmin,xmin),xmin,xmax);
  fit_graph_withOmegaA->SetParNames("A_{edm}","#omega_{edm}","#phi_{edm}","A_{vbo}","#omega_{vbo}","#phi_{vbo}","A_{g-2}","#omega_{a}","#phi_{a}");

  // EDM
  //fit_graph_withOmegaA->SetParameter(0,fakeAmp+amp_blind);
  //fit_graph_withOmegaA->SetParLimits(0,fakeAmp+amp_blind,fakeAmp+amp_blind);
  fit_graph_withOmegaA->SetParameter(1,freq);
  fit_graph_withOmegaA->SetParLimits(1,freq*0.9,freq*1.1);
  fit_graph_withOmegaA->SetParameter(2,fakePhase);
  fit_graph_withOmegaA->SetParLimits(2,fakePhase*0.9,fakePhase*1.1);
  // VBO
  //fit_graph_withOmegaA->SetParameter(3,avbo);
  //fit_graph_withOmegaA->FixParameter(3,avbo);
  fit_graph_withOmegaA->SetParameter(4,omega_vbo);
  fit_graph_withOmegaA->SetParLimits(4,omega_vbo*0.9,omega_vbo*1.1);
  //fit_graph_withOmegaA->SetParameter(5,phi_vbo*1.01);
  //fit_graph_withOmegaA->SetParLimits(5,phi_vbo*0.8,phi_vbo*1.2);
  // g-2
  //fit_graph_withOmegaA->SetParameter(6,newAmp);
  //fit_graph_withOmegaA->SetParLimits(6,newAmp,newAmp);
  fit_graph_withOmegaA->SetParameter(7,freq);
  fit_graph_withOmegaA->SetParLimits(7,freq*0.9,freq*1.1);
  fit_graph_withOmegaA->SetParameter(8,phi_a);
  fit_graph_withOmegaA->SetParLimits(8,phi_a*0.9,phi_a*1.1);
  
  fit_graph_withOmegaA->SetNpx(10000);
  fit_graph_withOmegaA->SetLineColor(kOrange+7);
  
  TGraphErrors* toFit_withOmegaA = (TGraphErrors*)testGraphWithOmegaA->Clone();
  toFit_withOmegaA->GetXaxis()->SetRangeUser(xmin,xmax);
  toFit_withOmegaA->GetYaxis()->SetRangeUser(-8e-4,8e-4);
  toFit_withOmegaA->Fit(fit_graph_withOmegaA);
  toFit_withOmegaA->Draw("AP");
  
  std::cout << "Chisq = " << fit_graph_withOmegaA->GetChisquare() << " NDF = " << fit_graph_withOmegaA->GetNDF() << "\n";

  double finalAmp3 = fit_graph_withOmegaA->GetParameter(0);
  finalAmp3 = (finalAmp3 / asymFactor) * gmagic;
  double finalDmu3 = getDmu(finalAmp3 - ( amp_blind * gmagic / asymFactor));
  std::cout << "finalAmp3 = " << fit_graph_withOmegaA->GetParameter(0) - amp_blind << " finalDmu3 = " << finalDmu3 << " inputDmu = " << TESTEDM << "\n";
  
  TPaveText* t1 = new TPaveText(50,6e-4,58,7.5e-4,"NB");
  t1->SetFillColor(0);
  t1->AddText(Form("#chi^{2}/dof = %f",fit_graph_withOmegaA->GetChisquare()/fit_graph_withOmegaA->GetNDF()));
  t1->AddText(Form("A_{vbo} = %.2f #pm %.2f #murad",fit_graph_withOmegaA->GetParameter(3)*1e6,fit_graph_withOmegaA->GetParError(3)*1e6));
  t1->Draw("SAME");  

  c1->SaveAs("testplots/fit_graph_withOmegaA.eps");

  // remove the blinding and refit
  TF1* fitUnblindedWithOmegaA = new TF1("fitUnblindedWithOmegaA",Form("-[0] * cos( [1] * (x+%.1f) + [2] ) - [3] * cos( [4] * (x+%.1f) + [5] - [6] * cos( [7] * (x+%.1f) + [8] ) )",xmin,xmin,xmin),xmin,xmax);
  fitUnblindedWithOmegaA->SetParNames("A_{edm}","#omega_{edm}","#phi_{edm}","A_{vbo}","#omega_{vbo}","#phi_{vbo}","A_{g-2}","#omega_{a}","#phi_{a}");
  fitUnblindedWithOmegaA->SetLineColor(kOrange+7);
  fitUnblindedWithOmegaA->SetNpx(10000);
  // EDM
  //fitUnblindedWithOmegaA->SetParameter(0,fakeAmp);
  //fitUnblindedWithOmegaA->SetParLimits(0,fakeAmp,fakeAmp);
  fitUnblindedWithOmegaA->SetParameter(1,freq);
  fitUnblindedWithOmegaA->SetParLimits(1,freq*0.99,freq*1.01);
  fitUnblindedWithOmegaA->SetParameter(2,fakePhase*0.99);
  fitUnblindedWithOmegaA->SetParLimits(2,fakePhase*0.95,fakePhase*1.05);
  // VBO
  //fitUnblindedWithOmegaA->SetParameter(3,avbo);
  //fitUnblindedWithOmegaA->FixParameter(3,avbo);
  fitUnblindedWithOmegaA->SetParameter(4,omega_vbo);
  fitUnblindedWithOmegaA->SetParLimits(4,omega_vbo*0.99,omega_vbo*1.01);
  fitUnblindedWithOmegaA->SetParameter(5,phi_vbo);
  fitUnblindedWithOmegaA->SetParLimits(5,phi_vbo*0.99,phi_vbo*1.01);
  // g-2
  //fitUnblindedWithOmegaA->SetParameter(6,newAmp);
  //fitUnblindedWithOmegaA->SetParLimits(6,newAmp,newAmp);
  fitUnblindedWithOmegaA->SetParameter(7,freq);
  fitUnblindedWithOmegaA->SetParLimits(7,freq*0.99,freq*1.01);
  fitUnblindedWithOmegaA->SetParameter(8,phi_a*1.0);
  fitUnblindedWithOmegaA->SetParLimits(8,phi_a*0.99,phi_a*1.01);
  
  c1->Clear();
  preBlindingWithOmegaA->SetTitle("Known EDM + vbo + g-2");
  preBlindingWithOmegaA->Fit(fitUnblindedWithOmegaA);
  preBlindingWithOmegaA->GetXaxis()->SetRangeUser(xmin,xmax);
  preBlindingWithOmegaA->Draw();
  c1->SaveAs("testplots/unblindedWithOmegaA.eps");

  double finalAmp2 = fitUnblindedWithOmegaA->GetParameter(0);
  finalAmp2 = (finalAmp2 / asymFactor) * gmagic;
  double finalDmu2 = getDmu(finalAmp2);
  std::cout << "finalAmp2 = " << fitUnblindedWithOmegaA->GetParameter(0) << " finalDmu2 = " << finalDmu2 << " inputDmu = " << TESTEDM << "\n";
  std::cout << "Chisq = " << fitUnblindedWithOmegaA->GetChisquare() << " NDF = " << fitUnblindedWithOmegaA->GetNDF() << "\n";
  


}


double getDelta(double dMu) {
  double eta = ((4 * mMuKg * c * dMu)/ (hbar * cm2m) );
  double tan_delta = (eta * beta) / (2 * aMu);
  double delta = atan(tan_delta);
  return delta;
}

double getDmu(double delta) {
  double tan_delta = tan(delta);
  double eta = (2 * aMu * tan_delta) / beta;
  double dMu = (eta * hbar * cm2m) / (4 * mMuKg * c);
  return dMu;
}


TH1* FFTHisto(TH1F* hist) {

  TH1* hist_FFT = 0;
  TVirtualFFT::SetTransform(0);
  hist_FFT = (TH1*)hist->FFT(hist_FFT,"MAG");

  int nbins = hist_FFT->GetNbinsX();
  double binw  = hist->GetBinWidth(1);
  double freq = 1/binw;

  double xfft1 = hist_FFT->GetXaxis()->GetXmin();
  double xfft2 = hist_FFT->GetXaxis()->GetXmax();

  TH1F* hist_FFT_scale = new TH1F("hfft","",nbins/2 + 1,xfft1,xfft2);
  for (int i(1); i<=( nbins/2 + 1); i++) {
    double y0 = (hist_FFT->GetBinContent(i) - hist_FFT->GetBinContent(nbins+1 - i));
    double y = sqrt(y0*y0);
    double ynew = y/sqrt(nbins);
    double x = hist_FFT->GetXaxis()->GetBinCenter(i);
    hist_FFT_scale->Fill(x,ynew);
  }

  hist_FFT->Delete();
  hist_FFT_scale->GetXaxis()->SetLimits(0, freq);

  return hist_FFT_scale;

}


TH1D* SetupFFT(TH1* h, double xmin, double xmax){
  double timeTot = xmax - xmin;
  double binW = h->GetBinWidth(1);
  int nBins = timeTot / binW;
  TH1D* hout = new TH1D("","",nBins, xmin, xmin + (nBins*binW));

  int binCount = 0;
  for (int i(0); i < h->GetXaxis()->GetNbins(); i++){
    if (h->GetBinCenter(i) < xmin ) continue;
    if (binCount > nBins) break;
    binCount++;
    double cont = h->GetBinContent(i);
    double err = h->GetBinError(i);
    hout->SetBinContent(binCount, cont);
    hout->SetBinError(binCount, err);
  }
  return hout;
}

TH1D* RescaleAxis(TH1* input, Double_t Scale) {
  int bins = input->GetNbinsX();
  TAxis* xaxis = input->GetXaxis();
  double* ba = new double[bins+1];
  xaxis->GetLowEdge(ba);
  ba[bins] = ba[bins-1] + xaxis->GetBinWidth(bins);
  for (int i = 0; i < bins+1; i++) {
    ba[i] *= Scale;
  }
  TH1D* out = new TH1D(input->GetName(), input->GetTitle(), bins, ba);
  for (int i = 0; i <= bins; i++) {
    out->SetBinContent(i, input->GetBinContent(i));
    out->SetBinError(i, input->GetBinError(i));
  }
  return out;
}
