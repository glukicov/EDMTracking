/*
 *    Blinders.h
 *
 *    Class to control blinding of the fit parameters in the omega_a and omega_p analyses
 *
 *    Created by Lawrence Gibbons
 *    08 January 2018
 *
 */

// use of typedefs of relevant class doesn't let me forward declare easily.
// #include <RandomLib/Random.hpp>
#include "/Users/gleb/software/EDMTracking/Blinding/RandomLib-1.10/include/RandomLib/Random.hpp"

#if !defined(BLINDING_BLINDERS_H)
#define BLINDING_BLINDERS_H
namespace blinding {


// class definition
class Blinders
{
public:
  
  enum fitType {kOmega_a, kOmega_p};
  
  // ---------- constructors ----------
  // -- Only fit type given, NO BLINDING IMPOSED IN THIS CASE
  Blinders( fitType type);
  
  // -- with fit type and string for hashing, blinding imposed
  Blinders( fitType type, const std::string& blindingString, double boxWidth = 24, double gaussWidth = 1 );
  
  // -- for blinding the shift direction in a systematic study
  Blinders( fitType type, unsigned int studyIndex, double nominalR, const std::string& blindingString, double boxWidth = 24, double gaussWidth = 1 );
  
  // ---------- do the blinding ----------
  // -- convert the fitting parameter to
  // -- omega_a fits:
  //      returns 2 * pi * 0.2291 MHz * (1 + (R +/- deltaR) *10^{-6})  (see E821 PRD)
  //      with R the relative precession frequency in ppm
  // -- omega_p fits:
  //      returns ??
  double paramToFreq( double blindedR );

  // -- for convenience, returns the reference value, eg  2 * pi * 0.2291 MHz for omega_a
  double referenceValue() { return m_reference; }
  
  // -- to allow double checking, return values of code
  
  // allow code to query the fit type under use
//  fitType fitType() { return m_type; }
  
  // destructor
  ~Blinders();
  
private:

  // -- issues a "loud" message to indicate that results will not be blinded
  void warning();
  
  // hide default constructor: force the Physicist to specify the behaviour...
  Blinders();
  
  // ----------------------- member items --------------------------
  
  //enum fitType m_type;
  
  // parameters for randomly throwing the blinding offset / sign
  double m_boxWidth; // default will be 24 ppm
  double m_gaussianWidth; // default will be 1 ppm
  
  // blinding values for central value
  double m_deltaR;
  double m_blindingSign;
  
  // additional blinding values for an indexed systematic study
  unsigned int m_systematicsIndex;
  double m_nominalR;
  double m_systematicsSign;
  
  // reference value
  double m_reference;
  
  // random generator
  RandomLib::Random* m_r;
};


} // end of blinding namespace

#endif
