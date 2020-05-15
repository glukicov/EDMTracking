/*
Copy a subset of a Tree to a new Tree if Track Quality or Vertex Quality is passed
   -Quality tracks are copied to a new Tree: QualityTracks
   -Quality vertices are copied to a new Tree: QualityVertices

   Author: Gleb Lukicov (g.lukicov@ucl.ac.uk) 13 Jan 2019
   Based on James's macros /gm2/app/users/jmott/analysis/TreePlotter

   run with (!note single quotes at the end and start!) :
   root 'ApplyTQ.C("input.txt", "output.txt")'
   where the text files contain full file-paths to input Trees
   and full path to be produced QualityTrees 

   e.g. 
         input.txt: 
                  /gm2/data/g2be/Production/Trees/Run1/trackRecoTrees_15921.root 
                  /gm2/data/g2be/Production/Trees/Run1/trackRecoTrees_15922.root 
                  etc... 
         output.txt: 
                  /gm2/data/g2be/Production/Trees/Run1/trackRecoTrees_15921.root 
                  /gm2/data/g2be/Production/Trees/Run1/trackRecoTrees_15922.root 
                  etc... 

*/
#include <time.h> // measure CPU time
#include <iostream>
#include <fstream>
#include "TFile.h"
#include "TTree.h"
using std::flush;
using std::cout;
using std::string;
using std::ifstream;
using std::vector;

// void ApplyTQ( string inputFilePath, string outputFilePath) {  // for a single file

// for many files with no arguments and input text files
void ApplyTQ(string inputTextFile, string outputTextFile) {

   //loop over lines in the input files
   ifstream infile1(inputTextFile);
   ifstream infile2(outputTextFile);
   string line;
   vector<string> inputFilePathVec;
   vector<string> outputFilePathVec;
   while (infile1 >> line) inputFilePathVec.push_back(line);
   while (infile2 >> line) outputFilePathVec.push_back(line);

   for (unsigned int i_file = 0; i_file < inputFilePathVec.size(); i_file++) {

      std::cout << "\nProcessing file " << i_file+1 << " out of " << inputFilePathVec.size() << "\n";
      std::cout << inputFilePathVec[i_file] << "\n";

      //get the input and output file names
      string inputFilePath=inputFilePathVec[i_file];
      string outputFilePath=outputFilePathVec[i_file];

      clock_t start, end;
      double cpu_time_used;
      start = clock();  // start counting

      //Open input file
      TFile *inputFile = new TFile(inputFilePath.c_str());
      //Get the tree
      TTree *inputTree = (TTree*)inputFile->Get("trackerNTup/tracker"); // name of the Tree
      //Check total number of entries
      Long64_t inputEntries = inputTree->GetEntries();
      cout << "Input entries: " << inputEntries << "\n";

      //Create a new file to clone trees into
      TFile *outputFile = new TFile(outputFilePath.c_str(), "recreate");
      // first clone with no entries
      TTree *outputTreeQT = inputTree->CloneTree(0); // QualityTracks tree
      TTree *outputTreeQV = inputTree->CloneTree(0); // QualityVerticies tree
      outputTreeQT->SetName("QualityTracks");
      outputTreeQV->SetName("QualityVertices");

      //resolve cuts
      bool passTrackQuality, passVertexQuality, passCandQuality;
      inputTree->SetBranchAddress("passTrackQuality", &passTrackQuality);
      inputTree->SetBranchAddress("passVertexQuality", &passVertexQuality);
      inputTree->SetBranchAddress("passCandidateQuality", &passCandQuality);
      int64_t passedQT = 0;
      int64_t passedQV = 0;

      int64_t tenth = int(inputEntries / 10); // find size of 10%
      int64_t tenth_counter = 1; //use for progress monitoring
      cout << "\nProgress at 0%..." << flush;

      // loop over all events and fill QT and QV
      for (int64_t entry = 0; entry < inputEntries; entry++) {

         inputTree->GetEntry(entry);

         // TQ is inclusive of CQ but just in case...
         if (passTrackQuality and passCandQuality) {
            outputTreeQT->Fill();
            passedQT++;
         } // TQ

         // VQ is inclusive of TQ but just in case..
         if (passVertexQuality and passTrackQuality and passCandQuality) {
            outputTreeQV->Fill();
            passedQV++;
         } // VQ

         // Printout progress based on the total entries remaining as 10%s 
         if (  entry + 1 == (tenth_counter * tenth) ) {
            cout <<  tenth_counter * 10 << "%..." << flush; // put +1 or face division by 0!
            tenth_counter++;
         } //progress

      } //entry

      cout << "\n\n"; // put new lines in printout

      // flush and save the tree, and close pointers
      outputTreeQT->AutoSave();
      outputTreeQV->AutoSave();
      delete outputFile;
      delete inputFile;

      end = clock(); // stop clock
      cpu_time_used = ( double(end) - double(start) ) / CLOCKS_PER_SEC;

      cout << "CPU time [s]: " << cpu_time_used << "\n";
      cout << "\nWrote quality tracks " << passedQT << "\n";
      cout << "Wrote quality vertices " << passedQV << "\n";
      cout << outputFilePathVec[i_file] << "\n";
   
   } // looping over file 

} // main
