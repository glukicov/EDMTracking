R__LOAD_LIBRARY(/gm2/app/users/scharity/Offline/Analysis/Blinding/dev_v9_30_00/build_slf6.x86_64/gm2util/lib/libgm2util_blinders.so)
#include "/gm2/app/users/scharity/Offline/Analysis/Blinding/dev_v9_30_00/srcs/gm2util/blinders/Blinders.hh"

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

#include <sstream>

// FFT fucntions (thanks tabitha)
TH1* FFTHisto(TH1F* hist);
TH1D* SetupFFT(TH1* h, double xmin, double xmax);
TH1D* RescaleAxis(TH1* input, Double_t Scale);
double getDelta(double dMu);

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

// edm to use for testing things, know things about this amplitude as we simulated it
double TESTEDM = d0 / 2.; 

// for plotting
std::string titleStr = "";
std::string title = "";
std::stringstream oss;
std::string plotname = "";

float xmin = 40.0;
float xmax = 60.0;

// blinding parameters
// blind by having a width of 2*d0 around R=10*d0 with broad gaussian tails
double R = 3.5; 
double boxWidth = 0.25;
double gausWidth = 0.7;
blinding::Blinders::fitType ftype = blinding::Blinders::kOmega_a;
blinding::Blinders getBlinded(ftype, "please change this default blinding stri", boxWidth, gausWidth);
// make a blinder for testing the unblinding procedure with a different plot
blinding::Blinders getBlindedTest(ftype,"testing blinding with py vs p plot", boxWidth, gausWidth);
//blinding::Blinders getBlindedTest(ftype,"do not use this string", boxWidth, gausWidth);

double blinded_edm_value( bool test = false ) {
  
  //
  // returns a blinded input edm value. returned dMu will be unphysical. it will be in the range of +- 3*d0 centred around 10*d0
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

void BlindedEDMAnalysis(){

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(1);

  // real data?
  bool realData = true;
  
  // testing with known edm?
  bool testFlag = false;
  
  if (testFlag) {
    std::cout << "========= not blinded ==========" << "\n";
    std::cout << "using test edm value of " << TESTEDM << " e.cm" << "\n";
  }

  // make a canvas
  TCanvas* c1 = new TCanvas();

  // get the data
  TFile* file = new TFile();
  TDirectoryFile* cuts;

  if (realData) {
    TFile* t1 = new TFile("../gm2TrackerEDMPlots_60hr.root");
    //    TFile* t1 = new TFile("../../../gm2analyses/ProductionScripts/produce/gm2trackerEDMPlots_ana_0_15987.00198.root");
    file = t1;
    titleStr = "60hr dataset";
    plotname = "h_thetay_vs_time_reco";
    TDirectoryFile* edm = (TDirectoryFile*)file->Get("EDM");
    TDirectoryFile* vol = (TDirectoryFile*)edm->Get("noVolEvents");
    cuts = (TDirectoryFile*)vol->Get("0<p<3100MeV");
    cuts->ls();
  }
  else {
    TFile* t2 = new TFile("VertexTailSelectorPlots_OriginalReco.root"); // SIM
    file = t2;
    titleStr = "gm2ringsim";
    plotname = "h_verticalMom_vs_time";
    TDirectoryFile* extrap = (TDirectoryFile*)file->Get("Extrapolation");
    TDirectoryFile* vertices = (TDirectoryFile*)extrap->Get("vertices");
    TDirectoryFile* stations = (TDirectoryFile*)vertices->Get("allStations");
    cuts = (TDirectoryFile*)stations->Get("pValue>0.005_and_noVolumesHit");
  }

  double dMu_blind = blinded_edm_value(testFlag);
  double delta_blind = getDelta(dMu_blind);
  
  // get the plot
  TH2F* h_pyVsT = (TH2F*)cuts->Get(plotname.c_str());
  
  //
  // make an N(t) plot to get the omega a phase
  //

  TProfile* p_pyVsT = (TProfile*)h_pyVsT->ProfileX();
  int nbinsx = p_pyVsT->GetNbinsX();
  int x1 = p_pyVsT->GetXaxis()->GetBinLowEdge(0);
  int x2 = p_pyVsT->GetXaxis()->GetBinUpEdge(nbinsx-1);
  TH1F* h_nt = new TH1F("h_nt","N(t) vs t",nbinsx,x1,x2);
  for (int ibin(0); ibin<p_pyVsT->GetNbinsX(); ibin++) {
    double ients = p_pyVsT->GetBinEntries(ibin);
    h_nt->SetBinContent(ibin,ients);
  }

  h_nt->GetXaxis()->SetRangeUser(xmin,xmax);

  TF1* ftmp = new TF1("ftmp",omega_func,xmin,xmax,6);

  ftmp->SetParNames("N","#gamma#tau","A","#omega","#phi","offset");

  ftmp->SetParameter(0,200e3);
  ftmp->SetParLimits(0,190e3,220e3);

  ftmp->SetParameter(1,64.4);
  ftmp->SetParLimits(1,63,65);

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

  ftmp->Draw("");
  c1->SaveAs("tmp.eps");
  
  //
  // add an oscillation with the blinded amplitude 
  //   

  // make a new histogram with the required oscillation
  double freq = getBlinded.referenceValue(); // omega
  double tan_amp = tan(delta_blind) / gmagic;
  double amp = 0.1 * atan(tan_amp); // 0.1 is asymmetry factor
  if (testFlag) {
    std::cout << "amp = " << amp << "; unboosted amp = " << delta_blind << "\n";
  }
  
  //
  // fake oscillation must be out of phase with g-2
  // make a guess of the phase from the number plot
  //
  
  double phi_a = ftmp->GetParameter(4); // get the phase from the n(t) fit
  double phase = phi_a + M_PI/2.;
  double off = xmin;

  TF1* edmOsc = new TF1("edmOsc",osc_func,xmin,xmax,4);
  edmOsc->SetParNames("A_{EDM blinded}","#omega_{a BNL}","#phi","offset");
  edmOsc->SetParameters(amp,freq,phase,off);
  edmOsc->SetLineColor(2);
  edmOsc->SetNpx(10000);

  // draw the function for checking (only if not blinded)
  if (testFlag) {
    edmOsc->GetXaxis()->SetRangeUser(xmin,xmax);
    edmOsc->Draw();
    c1->SaveAs("plots/edmOsc.eps");
  }
  
  TCanvas* c2 = new TCanvas();

  // make the fudged plot
  double minx = p_pyVsT->GetXaxis()->GetBinLowEdge(0);
  double maxx = p_pyVsT->GetXaxis()->GetBinUpEdge(nbinsx-1);
  TH1F* h_pyVsT_EDM = new TH1F("h_pyVsT_EDM","",nbinsx,minx,maxx);
  for (int ibin(0); ibin<p_pyVsT->GetNbinsX(); ibin++) {
    double binc = p_pyVsT->GetBinContent(ibin);
    double time = p_pyVsT->GetBinCenter(ibin);
    double new_binc = edmOsc->Eval(time);
    binc += new_binc;
    if (testFlag) {
      h_pyVsT_EDM->SetBinContent(ibin,new_binc);
    }
    else {
      h_pyVsT_EDM->SetBinContent(ibin,binc);
    }
    h_pyVsT_EDM->SetBinError(ibin,p_pyVsT->GetBinError(ibin));
  }
  h_pyVsT_EDM->GetXaxis()->SetTitle("time [#mus]");
  h_pyVsT_EDM->GetXaxis()->SetRangeUser(xmin,xmax);
  h_pyVsT_EDM->Draw("E");
  c2->SaveAs("plots/totalHistEDM.eps");

  //
  // FFT to see the frequencies
  //

  //Compute the transform and look at the magnitude of the output
  TH1 *h_FFT =0;
  TVirtualFFT::SetTransform(0);
  TH1D* h_pyVsT_EDM_s = SetupFFT(h_pyVsT_EDM, xmin, xmax);
  h_pyVsT_EDM_s->Draw("HIST");
  c2->SaveAs("plots/setup.eps");
  
  h_FFT = h_pyVsT_EDM_s->FFT(h_FFT,"MAG");
  TH1D* h_pyVsT_EDM_FFT = RescaleAxis(h_FFT,1./(xmax-xmin));
  h_pyVsT_EDM_FFT->SetTitle("FFT of total #theta_{y}(t) vs t (+ blinded input d_{#mu})");
  h_pyVsT_EDM_FFT->GetXaxis()->SetTitle("Frequency [MHz]");
  int nbins = h_pyVsT_EDM_FFT->GetXaxis()->GetNbins();
  h_pyVsT_EDM_FFT->GetXaxis()->SetRangeUser(0,h_pyVsT_EDM_FFT->GetXaxis()->GetBinUpEdge(nbins-1)/2);
  h_pyVsT_EDM_FFT->Draw("HIST");
  c2->SaveAs("plots/totalHistEDM_FFT.eps");


  //
  // for testing, make a fake EDM oscillation with known amplitude, and add in the VBO so we get something similar
  //
  
  double fakeDelta = getDelta(TESTEDM);
  double fakeTanAmp = tan(fakeDelta) / gmagic;
  double fakeAmp = 0.1 * atan(fakeTanAmp); // 0.1 is asymmetry factor
  std::cout << "fakeAmp = " << fakeAmp << "\n";

  double fakePhase = phase; // same as blinded EDM
  double fakeOff = xmin;

  TF1* fakeOsc = new TF1("fakeOsc",osc_func,xmin,xmax,4);
  fakeOsc->SetParNames("A","#omega_{a BNL}","#phi","offset");
  fakeOsc->SetParameters(fakeAmp,freq,fakePhase,fakeOff);
  fakeOsc->SetLineColor(2);
  fakeOsc->SetNpx(10000);
  fakeOsc->SetTitle("Toy MC, no blinding");
  fakeOsc->Draw();
  c2->SaveAs("testplots/fakeOsc.C");

  double avbo = 0.0002;
  double fvbo = 2.208;// MHz
  double omega_vbo = 2 * M_PI * fvbo;
  TF1* vboOsc = new TF1("vboOsc",osc_func,xmin,xmax,4);
  vboOsc->SetParNames("A_{vbo}","f_{vbo}","#phi","offset");
  vboOsc->SetParameters(avbo,omega_vbo,phi_a,xmin); // f in MHz
  vboOsc->Draw();
  fakeOsc->Draw("SAME");
  c2->SaveAs("testplots/vboOsc.C");

  std::cout << "freq = " << freq << "\n";

  // make the fudged plot
  double range = xmax - xmin;
  double binw = p_pyVsT->GetBinWidth(1);
  nbinsx = int(range/binw);
  int startBin = p_pyVsT->GetXaxis()->FindBin(xmin);
  std::cout << "startBin = " << startBin << " range = " << range << " binw = " << binw << " nbinsx = " << nbinsx << "\n";

  TProfile* h_fakeOsc = new TProfile("h_fakeOsc","",nbinsx,xmin,xmax,-0.02,0.02);
  for (int ibin(0); ibin<nbinsx; ibin++) {
    double time = p_pyVsT->GetBinCenter(ibin+startBin);
    double new_binc = fakeOsc->Eval(time);
    new_binc += vboOsc->Eval(time);
    
    // add some noise based on the bin errors
    // randomly get a number between new_binc +- bin_err
    double bin_err = p_pyVsT->GetBinError(ibin);
    //new_binc += gRandom->Uniform(-bin_err,bin_err);
    //    new_binc += (gRandom->Rndm()*0.0001); // add some random noise
    h_fakeOsc->SetBinContent(ibin,new_binc);
    h_fakeOsc->SetBinError(ibin,bin_err);
    h_fakeOsc->SetBinEntries(ibin,p_pyVsT->GetBinEntries(ibin));
  }
  
  h_fakeOsc->GetXaxis()->SetTitle("time [#mus]");
  h_fakeOsc->GetXaxis()->SetRangeUser(xmin,xmax);
  h_fakeOsc->SetTitle(Form("Known input EDM = %.2e e.cm + VBO + noise",TESTEDM));
  h_fakeOsc->Draw("EP");
  c2->SaveAs("testplots/fakeOscHist.C");

  //
  // test the blinding with this plot -- add an oscillation with blinded amplitude on top of the plot
  //

  double atest = blinded_test_osc();
  TF1* testOsc = new TF1("testOsc",osc_func,xmin,xmax,4);
  testOsc->SetParNames("A_{blinded}","freq","phase","offset");
  testOsc->SetParameters(atest,freq,phase,xmin);
  testOsc->SetLineColor(2);
  testOsc->SetNpx(10000);

  nbinsx = h_fakeOsc->GetNbinsX();
  minx = h_fakeOsc->GetXaxis()->GetBinLowEdge(0);
  maxx = h_fakeOsc->GetXaxis()->GetBinUpEdge(nbinsx-1);

  TH1F* h_fakeOsc_blind = new TH1F("h_fakeOsc_blind","",nbinsx,minx,maxx);
  for (int ibin(0); ibin<h_fakeOsc->GetNbinsX(); ibin++) {
    double binc = h_fakeOsc->GetBinContent(ibin);
    double time = h_fakeOsc_blind->GetBinCenter(ibin);
    double new_binc = testOsc->Eval(time);
    binc += new_binc;
    h_fakeOsc_blind->SetBinContent(ibin,binc);
    h_fakeOsc_blind->SetBinError(ibin,h_fakeOsc->GetBinError(ibin));
  }
  h_fakeOsc_blind->GetXaxis()->SetTitle("time [#mus]");
  h_fakeOsc_blind->GetXaxis()->SetRangeUser(xmin,xmax);
  TH1F* h_fakeOsc_blind_fit = (TH1F*)h_fakeOsc_blind->Clone();
  h_fakeOsc_blind->SetTitle("MC osc + blinded EDM");
  h_fakeOsc_blind->Draw("P");
  c2->SaveAs("testplots/h_fakeOsc_blind.C");
  
  // now FFT this plot -- should see two frequencies

  TH1 *h_FFT_test =0;
  TVirtualFFT::SetTransform(0);
  TH1D* h_fakeOsc_blind_s = SetupFFT(h_fakeOsc_blind, xmin, xmax);
  h_fakeOsc_blind_s->Draw("HIST");
  c2->SaveAs("testplots/setup_fakeOsc.C");
  
  h_FFT_test = h_fakeOsc_blind_s->FFT(h_FFT_test,"MAG");
  TH1D* h_fakeOsc_blind_FFT = RescaleAxis(h_FFT_test,1./(xmax-xmin));
  h_fakeOsc_blind_FFT->SetTitle("FFT of fake oscillation + blind EDM osc + fake VBO");
  h_fakeOsc_blind_FFT->GetXaxis()->SetTitle("Frequency [MHz]");
  nbins = h_fakeOsc_blind_FFT->GetXaxis()->GetNbins();
  //  h_fakeOsc_blind_FFT->GetXaxis()->SetRangeUser(0,h_fakeOsc_blind_FFT->GetXaxis()->GetBinUpEdge(nbins-1)/2);
  h_fakeOsc_blind_FFT->GetXaxis()->SetRangeUser(0,3.5);
  h_fakeOsc_blind_FFT->Draw("HIST");
  c2->SaveAs("testplots/fakeOscBlind_FFT.C");

  // fit the oscillation for the frequencies we know we've put in
  // the EDM peak now has an unknown amplitude so we should let that float

  //  return (-amp * cos(freq * time + phi));
  std::string totalFitFunc = "-[0] * cos([1] * (x + [2]) + [3]) - [4] * cos([5] * (x + [2]) + [6])" ;
  TF1* fakeOscFit = new TF1("fakeOscFit",totalFitFunc.c_str(),xmin,xmax);

  // offset
  fakeOscFit->SetParameter(2,xmin);
  fakeOscFit->FixParameter(2,xmin);
  
  // first oscillation -- vbo
  
  //fakeOscFit->SetParameter(0,0.00025);
  //  fakeOscFit->FixParameter(0,0.0001);
  //fakeOscFit->SetParLimits(0,0.0001,0.0003);
  
  // we know the frequency from the FFT
  fakeOscFit->SetParameter(1,omega_vbo);
  fakeOscFit->FixParameter(1,omega_vbo);
  
  fakeOscFit->SetParameter(3,phi_a);
  fakeOscFit->FixParameter(3,phi_a);

  // now fit the edm peak. we know freq and phase, let A float
  fakeOscFit->SetParameter(5,freq);
  fakeOscFit->FixParameter(5,freq);
  
  fakeOscFit->SetParameter(6,phase);
  fakeOscFit->FixParameter(6,phase);
  //fakeOscFit->SetParLimits(6,phase*0.9,phase*1.1);
  
  // maybe give amplitude a guess and limits
  //fakeOscFit->SetParameter(4,0.001);
  //fakeOscFit->SetParLimits(4,0.0004,0.0015);
						
  fakeOscFit->SetLineColor(2);
  fakeOscFit->SetNpx(10000);

  h_fakeOsc_blind->Fit(fakeOscFit);
  
  h_fakeOsc_blind->GetXaxis()->SetRangeUser(xmin,xmin+20);
  h_fakeOsc_blind->SetTitle("Fit to blinded total hist");
  h_fakeOsc_blind->Draw("E");
  c2->SaveAs("testplots/fakeOscFit.C");

  // subtract the blinded oscillation from this histogram and check we recover the original histogram - refitting should give us what we put in
  TH1F* h_fakeOsc_unblinded = new TH1F("h_fakeOsc_unblinded","",nbinsx,minx,maxx);
  std::cout << "nbinsx = " << nbinsx << "\n";
  for (int ibin(0); ibin<h_fakeOsc_blind->GetNbinsX(); ibin++) {
    double binc = h_fakeOsc_blind->GetBinContent(ibin);
    double time = h_fakeOsc_unblinded->GetBinCenter(ibin);
    binc -= testOsc->Eval(time); // subtract the blinding
    h_fakeOsc_unblinded->SetBinContent(ibin,binc);
    h_fakeOsc_unblinded->SetBinError(ibin,h_fakeOsc_blind->GetBinError(ibin));
  }

  h_fakeOsc_unblinded->GetXaxis()->SetTitle("time [#mus]");
  //  h_fakeOsc_unblinded->GetXaxis()->SetRangeUser(xmin,xmin+20);
  //  h_fakeOsc_unblinded->SetTitle("Unblinded test");

  // fit it
  TF1* unblindedFit = new TF1("unblindedFit",totalFitFunc.c_str(),xmin,xmax);

  unblindedFit->SetParameter(2,xmin);
  unblindedFit->FixParameter(2,xmin);
  
  // first oscillation -- vbo
  
  unblindedFit->SetParameter(0,fakeOscFit->GetParameter(0));
  unblindedFit->FixParameter(0,fakeOscFit->GetParameter(0));
  
  // we know the frequency from the FFT
  unblindedFit->SetParameter(1,fakeOscFit->GetParameter(1));
  unblindedFit->FixParameter(1,fakeOscFit->GetParameter(1));
  
  unblindedFit->SetParameter(3,fakeOscFit->GetParameter(3));
  unblindedFit->FixParameter(3,fakeOscFit->GetParameter(3));
  //unblindedFit->SetParLimits(2,phi_a*0.9,phi_a*1.1);

  // now fit the edm peak. we know freq and phase, let A float
  unblindedFit->SetParameter(5,freq);
  unblindedFit->FixParameter(5,freq);
  
  unblindedFit->SetParameter(6,phase);
  unblindedFit->FixParameter(6,phase);

  unblindedFit->SetParameter(4,3e-6);
  unblindedFit->SetParLimits(4,1e-6,4e-6);
  
  unblindedFit->SetLineColor(2);
  unblindedFit->SetNpx(10000);
  
  //  h_fakeOsc_unblinded->GetXaxis()->SetRangeUser(xmin,xmin+20);
  h_fakeOsc_unblinded->Fit(unblindedFit);
  h_fakeOsc_unblinded->SetTitle("Fit to total osc, unblinded");
  h_fakeOsc_unblinded->Draw("E");
  c2->SaveAs("testplots/h_fakeOsc_unblinded.C");

  // fit the 'before' histo to compare
  TF1* trueFit = new TF1("trueFit",totalFitFunc.c_str(),xmin,xmax);
  trueFit->SetParameters(avbo,omega_vbo,xmin,phi_a,fakeAmp,freq,phase);
  trueFit->FixParameter(0,avbo);
  trueFit->FixParameter(1,omega_vbo);
  trueFit->FixParameter(2,xmin);
  trueFit->FixParameter(3,phi_a);
  trueFit->FixParameter(4,fakeAmp);
  trueFit->FixParameter(5,freq);
  trueFit->FixParameter(6,phase);
  trueFit->SetLineColor(2);
  trueFit->SetNpx(10000);

  h_fakeOsc->SetTitle("MC histo + fit, before blinding");
  h_fakeOsc->Fit(unblindedFit);
  h_fakeOsc->Draw("E");
  c2->SaveAs("testplots/h_fakeOsc_before_fit.C");
  
  //  return 0;

}


double getDelta(double dMu) {
  double eta = ((4 * mMuKg * c * dMu)/ (hbar * cm2m) );
  double tan_delta = (eta * beta) / (2 * aMu);
  double delta = atan(tan_delta);
  return delta;
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
