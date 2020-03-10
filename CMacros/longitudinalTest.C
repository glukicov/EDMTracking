// Author: James
// Modified by Gleb (8 March 2020)

void longitudinalTest() {

  TFile f("longitudinalTest.root", "recreate");
  f.cd();

  //------ constants
  int n_inject = 2000000;

  // plots limits
  double start_time = 0; // us
  double end_time = 100; //  us
  double bin_w = 0.150; // 150 ns
  int bin_n = 4000;
  double angle_bin_max = 1.2; // arbitrary

  // expected parameters (magic/simulation)
  double lifetime_magic = 64.94; // us
  double phase_magic = 6.295; // rad
  double asym_magic = -0.386;
  double omega_magic = 1.439; // MhZ

  // Amplitudes
  double A_edm = 1.0; // large EDM 
  double A_bz  = 0.5*A_edm;  // small B_z/A_u

  //smearing 
  double angRes = 0.01;
  TRandom3* rng = new TRandom3(12345);
  gRandom->SetSeed(12345);

  //---------

  //generative functions
  TF1* fWiggle = new TF1("fWiggle", "[0]*exp(-x/[1])*(1+[2]*cos([3]*x+[4]))", start_time, end_time);
  fWiggle->SetParameters(1, lifetime_magic, asym_magic, omega_magic, phase_magic); fWiggle->SetNpx(10000);

  // vertical angle oscillation
  TF1* fVertical = new TF1("fVertical", "[0]*cos([2]*x+[3]) + [1]*sin([2]*x+[3])", 0, end_time);
  fVertical->SetParameters(A_bz, A_edm, omega_magic, phase_magic); fVertical->SetNpx(10000);

  // Do a pseudo experiment and fit for frequency
  TH1F* hitTimes = new TH1F("hitTimes", ";Time [us];Counts", bin_n, 0, bin_w * bin_n);
  for (int i = 0 ; i < n_inject; i++) {
    double t = fWiggle->GetRandom();
    hitTimes->Fill(t);
  }

  //fit for the wiggle
  TF1* fitFunc = new TF1("fitFunc", "[0]*exp(-x/[1])*(1+[2]*cos([3]*x+[4]))", start_time, end_time);
  fitFunc->SetParameters(n_inject * bin_w / lifetime_magic, lifetime_magic, asym_magic, omega_magic, phase_magic);
  hitTimes->Fit(fitFunc, "RLQ");
  double period = TMath::TwoPi() / fitFunc->GetParameter(3);
  double offset = fitFunc->GetParameter(4) / fitFunc->GetParameter(3);
  double fitLifetime = fitFunc->GetParameter(1);

  // Now remake plots modulo period
  TH1F* hitTimesMod = new TH1F("hitTimesMod", ";Time % #omega_{a} Period [us];Counts", bin_n, -period, period);
  TH2F* hitAngleMod = new TH2F("hitAngleMod", ";Time % #omega_{a} Period [us];Angle [arb. units]", bin_n, -period, period, 1000, -angle_bin_max, angle_bin_max);
  TH2F* hitAngleModRefl = new TH2F("hitAngleModRefl", ";Time % #omega_{a} Period [us];Angle [arb. units]", 2000, 0, period, 1000, -angle_bin_max, angle_bin_max);

  //Re-weighting
  for (int i = 0 ; i < n_inject; i++) {
    double t = fWiggle->GetRandom();
    double ang = rng->Gaus(fVertical->Eval(t), angRes);
    double tMod = fmod(t - offset, 2 * period) - period;
    double weight = exp(tMod / fitLifetime);
    hitTimesMod->Fill(tMod, weight);
    hitAngleMod->Fill(tMod, ang, weight);
    if (tMod > 0) hitAngleModRefl->Fill(tMod, ang, weight);
    else         hitAngleModRefl->Fill(-tMod, ang, weight);
  }

  TCanvas* time = new TCanvas("time", "time");
  hitTimesMod->Draw();

  TCanvas* angles = new TCanvas("angles", "angles");
  hitAngleMod->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max );
  hitAngleMod->Draw("COLZ");

  TCanvas* anglesRefl = new TCanvas("anglesRefl", "anglesRefl");
  hitAngleModRefl->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max );
  hitAngleModRefl->Draw("COLZ");

  TCanvas* anglesProf = new TCanvas("anglesProf", "anglesProf");
  TProfile* hitAngleModProf = hitAngleMod->ProfileX("hitAngleModProf");
  hitAngleModProf->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max );
  hitAngleModProf->Draw("COLZ");

  TCanvas* anglesProfRefl = new TCanvas("anglesProfRefl", "anglesProfRefl");
  TProfile* hitAngleModReflProf = hitAngleModRefl->ProfileX("hitAngleModReflProf");
  hitAngleModReflProf->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max );
  hitAngleModReflProf->Draw("HISTP");

  TF1* fSinCos = new TF1("fSinCos", "[0]*cos([2]*x)+[1]*sin([2]*x)", 0, period);
  fSinCos->SetParameter(0, 0.5);
  fSinCos->SetParameter(1, 0.5);
  fSinCos->FixParameter(2, omega_magic);
  fSinCos->SetNpx(10000);
  fSinCos->SetLineColor(2);
  hitAngleModReflProf->Fit(fSinCos, "RN");
  cout << "Input: " << endl;
  cout << "A_bz = " << A_bz << "\t A_edm= " << A_edm << endl;
  cout << "Fit: " << endl;
  cout << "A_bz = " << fSinCos->GetParameter(0) << "\t A_edm = " << fSinCos->GetParameter(1) << endl;

  TCanvas* cPlot = new TCanvas("cPlot", "cPlot");
  cPlot->SetTopMargin(0);
  cPlot->SetBottomMargin(0);
  cPlot->Divide(1, 2, 0, 0);
  cPlot->cd(1);
  gPad->SetBottomMargin(0);
  gPad->SetTopMargin(0.17);
  TF1* fCos = new TF1("fCos", "[0]*cos([1]*x)", 0, period);
  fCos->SetParameter(0, A_bz );
  fCos->FixParameter(1, TMath::TwoPi() / period);
  fCos->SetNpx(10000);
  fCos->SetLineColor(2);
  TF1* fSin = new TF1("fSin", "[0]*sin([1]*x)", 0, period);
  fSin->SetParameter(0, A_edm );
  fSin->FixParameter(1, TMath::TwoPi() / period);
  fSin->SetNpx(10000);
  fSin->SetLineColor(4);
  fCos->SetTitle(";;Angle [arb. units]");
  fCos->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max );
  fCos->GetYaxis()->SetTitleSize(0.07);
  fCos->GetYaxis()->SetTitleOffset(0.5);
  fCos->GetYaxis()->SetLabelSize(0.07);
  fCos->GetYaxis()->CenterTitle();
  fCos->Draw();
  fSin->Draw("SAME");
  fVertical->SetParameter(3, 0);
  fVertical->Draw("SAME");

  cPlot->cd(2);
  gPad->SetTopMargin(0);
  gPad->SetBottomMargin(0.17);
  hitAngleModReflProf->GetYaxis()->SetTitle("Angle [arb. units]");
  hitAngleModReflProf->SetStats(0);
  hitAngleModReflProf->GetYaxis()->CenterTitle();
  hitAngleModReflProf->GetYaxis()->SetTitleSize(0.07);
  hitAngleModReflProf->GetYaxis()->SetTitleOffset(0.5);
  hitAngleModReflProf->GetYaxis()->SetLabelSize(0.07);
  hitAngleModReflProf->GetXaxis()->SetLabelSize(0.07);
  hitAngleModReflProf->GetXaxis()->SetTitleSize(0.07);
  hitAngleModReflProf->Draw();

  cPlot->SaveAs("../fig/AzimuthalField.png");

  time->Write();
  angles->Write();
  anglesRefl->Write();
  anglesProf->Write();
  anglesProfRefl->Write();
  cPlot->Write();

  f.Write();
  f.Close();

  cout << "final figure saved!\n";

}