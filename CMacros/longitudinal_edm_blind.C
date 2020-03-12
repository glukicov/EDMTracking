// Author: James Mott 
// Modified by Gleb (8 March 2020) 

// A toy model to blind the EDM signal to do a fit for B_z 
// See Gleb's DocDB:XXXXX for details (to be uploaded ~Mid March 2020)
// Main implementation is in Python
// e.g. for simulation see https://github.com/glukicov/EDMTracking/blob/master/JupyterNB/Bz_sim.ipynb

void longitudinal_edm_blind() {

  int n_inject = int(1013836);
  
  // expected parameters (magic/simulation)
  double lifetime_magic = 64.04; // us
  double phase_magic = 6.295; // rad
  double asym_magic = -0.386;
  double omega_magic = 1.439; // MhZ
  double c_magic = 0.042; //mrad vertical angle offset 

  // Data-derived 
  double g2period = TMath::TwoPi() / omega_magic;  // ~4.365 us 
  double phase_offset = phase_magic / omega_magic; // adjust by phase us 
  double lifetime_weight = lifetime_magic; // can also be extracted from the fit to w_a 

    // plots limits
  double start_time = 4.5; // us
  double end_time = 100; //  us
  int bin_n = 4000;
 
  // Amplitudes
  double A_edm = 0.14; // large EDM 
  double A_bz  = -0.04;  // small B_z/A_u
  double angle_bin_max = max(A_edm, A_bz)*1.6; //just for plotting - min/max

  //smearing 
  double angRes = 0.01;
  TRandom3* rng = new TRandom3(12345);
  gRandom->SetSeed(12345);

  ofstream dump;
  dump.open ("../DATA/misc/C_dump.txt");
  dump << "t" << ", " << "ang"  << ", " << "tMod" << ", " << "weight" << "\n";
  //---------end of const

  //----functions 

  //generative function (can be taken from data)
  TF1* fWiggle = new TF1("fWiggle", "[0]*exp(-x/[1])*(1+[2]*cos([3]*x+[4]))", start_time, end_time);
  fWiggle->SetParameters(1, lifetime_magic, asym_magic, omega_magic, phase_magic); 

  // B_z with magic parameters 
  TF1* f_bz = new TF1("f_bz", "[0]*cos([1]*x+[2])+[3]", 0, g2period);
  f_bz->SetParameter(0, A_bz ); f_bz->FixParameter(1, omega_magic); f_bz->FixParameter(2, phase_magic); f_bz->FixParameter(3, c_magic); f_bz->SetLineColor(2); // red 
  
  // EDM with magic parameters 
  TF1* f_edm = new TF1("f_edm", "[0]*sin([1]*x+[2])+[3]", 0, g2period);
  f_edm->SetParameter(0, A_edm );  f_edm->FixParameter(1, omega_magic);f_edm->FixParameter(2, phase_magic); f_edm->FixParameter(3, c_magic); f_edm->SetLineColor(4); // blue

  // vertical angle oscillation (generative function!/can be taken from data, also used for visualisation)
  TF1* fVertical = new TF1("fVertical", "[0]*cos([2]*x-[3]) + [1]*sin([2]*x-[3])+[4]", 0, end_time);
  fVertical->SetParameters(A_bz, A_edm, omega_magic, phase_magic, c_magic);  fVertical->SetLineColor(1);  fVertical->SetLineStyle(3);  // greed-dashed

  // The convolution function (we fix omega and phase and fit for A_bz, c, and A_edm_BLINDED - safe to show)
  TF1* f_conv = new TF1("f_conv", "[0]*cos([2]*x+[3])+[1]*sin([2]*x+[3])+[4]", 0, g2period);
  f_conv->FixParameter(2, omega_magic); f_conv->FixParameter(3, phase_magic); f_conv->SetLineColor(6); // purple 
  //-----------end of func

  //---- plots 
  TFile f("../DATA/misc/toy_blind.root", "recreate"); f.cd();
  TH1F* hitTimesMod = new TH1F("hitTimesMod", ";Time % #omega_{a} g2period [us];Counts", bin_n, -g2period, g2period);
  TH2F* hitAngleMod = new TH2F("hitAngleMod", ";Time % #omega_{a} g2period [us];Angle [mrad]", bin_n, -g2period, g2period, bin_n/4, -angle_bin_max, angle_bin_max);
  // TH2F* hitAngleModRefl = new TH2F("hitAngleModRefl", ";Time % #omega_{a} g2period [us];Angle [mrad]", bin_n/2, -g2period*2, g2period*2, bin_n/4, -angle_bin_max, angle_bin_max);
  TH2F* hitAngleModRefl = new TH2F("hitAngleModRefl", ";Time % #omega_{a} g2period [us];Angle [mrad]", bin_n/2, 0, g2period, bin_n/4, -angle_bin_max, angle_bin_max);
  //---end of pots 

  // -----MAIN-----

  //Re-weighting
  for (int i = 0 ; i < n_inject; i++) {
    double t = fWiggle->GetRandom(); // can be taken from sim 
    double ang = rng->Gaus(fVertical->Eval(t), angRes); // can  be taken from sim  
    double tMod = fmod(t - phase_offset, 2 * g2period) - g2period;
    double weight = exp(tMod / lifetime_weight);
    // cout << "tMod " << tMod << " weight " << weight << "\n";
    dump << t << ", " << ang  << ", " << tMod << ", " << weight << "\n";
    hitTimesMod->Fill(tMod, weight);
    hitAngleMod->Fill(tMod, ang, weight);
    if (tMod > 0) hitAngleModRefl->Fill(tMod, ang, weight);
    else         hitAngleModRefl->Fill(-tMod, ang, weight);
  }
  
  //We will use profile (hitAngleModReflProf) for fits of convolution function
  TProfile* hitAngleModProf = hitAngleMod->ProfileX("hitAngleModProf");
  TProfile* hitAngleModReflProf = hitAngleModRefl->ProfileX("hitAngleModReflProf");
    
  // Now this is where the fit happens (i.e. IT IS SAFE TO FIT TO THIS PROFILE)
  hitAngleModReflProf->Fit(f_conv, "RN"); // range [0, g2period], N=no_draw 
  cout << "\nTruth: A_bz = " << A_bz << "\t A_edm= " << A_edm << "\nConv.  : A_bz = " << f_conv->GetParameter(0) << "\t A_edm = " << f_conv->GetParameter(1) << "\n\n";

  //--- end of main 

  //save the final canvas
  TCanvas* cPlot = new TCanvas("cPlot", "cPlot"); cPlot->SetTopMargin(0); cPlot->SetBottomMargin(0); cPlot->Divide(1, 2, 0, 0); cPlot->cd(1); gPad->SetBottomMargin(0); gPad->SetTopMargin(0.17);
  //Draw EDM, B_z, theta 
  f_bz->Draw(); f_edm->Draw("SAME"); fVertical->Draw("SAME");
  f_bz->SetTitle(";;Angle [mrad]"); f_bz->GetYaxis()->SetRangeUser(-angle_bin_max , angle_bin_max ); f_bz->GetYaxis()->SetTitleSize(0.07); f_bz->GetYaxis()->SetTitleOffset(0.5); f_bz->GetYaxis()->SetLabelSize(0.07); f_bz->GetYaxis()->CenterTitle();
  // the resultant plot 
  cPlot->cd(2); gPad->SetTopMargin(0); gPad->SetBottomMargin(0.17); hitAngleModReflProf->GetYaxis()->SetTitle("Angle [mrad]"); hitAngleModReflProf->SetStats(0); hitAngleModReflProf->GetYaxis()->CenterTitle(); hitAngleModReflProf->GetYaxis()->SetTitleSize(0.07); hitAngleModReflProf->GetYaxis()->SetTitleOffset(0.5); hitAngleModReflProf->GetYaxis()->SetLabelSize(0.07); hitAngleModReflProf->GetXaxis()->SetLabelSize(0.07); hitAngleModReflProf->GetXaxis()->SetTitleSize(0.07);
  hitAngleModReflProf->Draw();
  //save final plot 
  cPlot->SaveAs("../fig/toy_blind.png"); f.Write(); f.Close();
  cout << "final figure saved, done!\n";

}