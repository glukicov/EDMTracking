/*
 *  Blinders.cpp
 *  Blinders
 *
 *  Source code for the Muon g-2 analysis blinding utility
 *  Author:  Lawrence Gibbons, Cornell University
 *
 *
 */

#include <iostream>
#include <cmath>
#include <openssl/md5.h>
#include <string>

#include "Blinders.hh" 
#include "RandomLib/NormalDistribution.hpp" 

const double k_pi = 3.14159265358979323846264338327950288419716939937510582;
const double k_omega_a_ref = 0.2291; // MHz

const double k_omega_p_ref =  2.79284734462 * 7.622593285      * 1.5;
//                            (mu_p / mu_N) * (mu_N/h) [MHz/T] * B [T]

// convert the MD5 character digest (uint8) into seed words for the random generator (uint32)
const int k_seedlength = MD5_DIGEST_LENGTH / 4;

// specify the relative precision that R represents, in this implementation we are specifying shifts in ppm
const double k_precisionR = 1e-6;

using namespace blinding;

// Code in extern to enable python access
#ifdef __cplusplus
extern "C" {
#include "python_header.h" 
  }
#endif

// ******************************************************************************************************
// ******************************************************************************************************
Blinders::Blinders( Blinders::fitType ftype ) :
m_deltaR( 0 ),
m_blindingSign( 1 ),
m_nominalR( 0 ),
m_systematicsSign( 1 ),
m_r( 0 )
{
  // choose the appropriate reference value
  if ( ftype == Blinders::kOmega_a ) {
    m_reference = 2 * k_pi * k_omega_a_ref;
  } else if ( ftype == Blinders::kOmega_p ) {
    m_reference = 2 * k_pi * k_omega_p_ref;
  } else {
    throw std::runtime_error("Invalid fit type in Blinders constructor");
  }
  
  // no blinding will happen, so issue a warning
  warning();
} // end of constructor



// ******************************************************************************************************
// ******************************************************************************************************
// constructor for the blinding the standard analysis of the central value
// all the work is done in the most general constructor with the appropriate nominal values
Blinders::Blinders( Blinders::fitType ftype, const std::string& blindingString, double boxWidth, double gaussWidth ) :
Blinders( ftype, 0, 0, blindingString, boxWidth, gaussWidth )
{
  std::cout << " + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ +" << std::endl;
  std::cout << " +                                                                      +" << std::endl;
  std::cout << " +           You have chose to blind your fitting according to          +" << std::endl;
  std::cout << " +                omega_ref * (1 + (R +/- deltaR) *10^{-6})             +" << std::endl;
  std::cout << " +                                                                      +" << std::endl;
  std::cout << " + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ +" << std::endl;
}



// ******************************************************************************************************
// ******************************************************************************************************

// constructor that allows you to blind the sign within a systematic study.
// The constructor for blinding the standard analysis can be obtained from
// this one with studyIndex = 0
Blinders::Blinders( fitType ftype, unsigned int studyIndex, double nominalR, const std::string& blindingString,  double boxWidth, double gaussWidth  ) :
m_boxWidth(boxWidth),
m_gaussianWidth(gaussWidth),
m_systematicsIndex( studyIndex ),
m_nominalR( nominalR ),
m_systematicsSign( 1 )
{
  if ( ftype == Blinders::kOmega_a ) {
    m_reference = 2 * k_pi * k_omega_a_ref;
  } else if ( ftype == Blinders::kOmega_p ) {
    m_reference = 2 * k_pi * k_omega_p_ref;
  } else {
    throw std::runtime_error("Invalid fit type in Blinders constructor");
  }
  
  //create the seed list from the input string
  union SeedHash {
    unsigned char digest[MD5_DIGEST_LENGTH];
    std::uint32_t seeds[k_seedlength];
  };
  SeedHash sh;
  MD5( (const unsigned char *)blindingString.c_str(), blindingString.length(), (unsigned char *)&sh.digest );
  
  // Get and seed the random number generator
  m_r = new RandomLib::Random(sh.seeds, sh.seeds+k_seedlength );
#ifdef DEBUG_BLINDING
  std::cout << "Created random generator with seeds " << m_r->SeedString() << std::endl;
#endif
  
  // Determine the blinding parameters for the central study
  // First, the obfuscating sign
  m_blindingSign = 1;
  if ( m_r->Float() < 0.5 ) {
    m_blindingSign = -1;
  }
  // Next, the blinding shift itself
  // -- throw a flat box
  double effectiveBoxWidth = 2 * m_boxWidth + sqrt(k_pi * 2) * m_gaussianWidth;
  double boxVal = m_r->Float() * effectiveBoxWidth - (effectiveBoxWidth/2);
  m_deltaR = boxVal;
  
  // -- we have entered into guassian territory that allows the smearing range to be unbounded
  if ( fabs(boxVal) > m_boxWidth ) {
    RandomLib::NormalDistribution<> guassGen;
    double guassVal = guassGen( *m_r, 0, m_gaussianWidth );
    double absR = m_boxWidth + fabs(guassVal);
    if ( boxVal > 0 ) {
      m_deltaR = absR;
    } else {
      m_deltaR = -absR;
    }
  } // end of guassian smearing block
#ifdef DEBUG_BLINDING
  std::cout << "Have created blinding parameters" << std::endl;
  std::cout << "   deltaR = " << m_deltaR       << std::endl;
  std::cout << "   blSign = " << m_blindingSign << std::endl;
#endif
  
  // if we are blinding the sign of the shift of a systematics study relative to a nominal value,
  // so now determine the blinding sign of the systematic shift
  if ( m_systematicsIndex > 0 ) {
    double useThisRandom = -1;
    // skip to a random number that is at count m_systematicsIndex later in the generated sequence
    for ( unsigned int index = 0; index < m_systematicsIndex; ++index ) {
      useThisRandom = m_r->Float();
    }
    m_systematicsSign = 1;
    if ( useThisRandom < 0.5 ) m_systematicsSign = -1;
    
    std::cout << " + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ +" << std::endl;
    std::cout << " +                                                                                       +" << std::endl;
    std::cout << " +           You have chose to blind your systematic study fitting according to          +" << std::endl;
    std::cout << " +            omega_ref * (1 + (R +/- deltaR)_nominal +/- deltaR_syst) *10^{-6})         +" << std::endl;
    std::cout << " +                                                                                       +" << std::endl;
    std::cout << " + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ +" << std::endl;
  } else {
    m_systematicsSign = 1;
  }
}  // end of constructor

// ******************************************************************************************************
// ******************************************************************************************************
Blinders::~Blinders() {
  if ( m_r != 0 ) delete m_r;
}



// ******************************************************************************************************
// ******************************************************************************************************
double
Blinders::paramToFreq( double blindedValue )
{
  double freq = 0;
  
  // if we are not doing a blinded systematic study, the blinded R has been passed in
  if ( m_systematicsIndex == 0 ) {
    double unblinded_R = blindedValue - (m_blindingSign * m_deltaR);
    freq = m_reference * ( 1 + (unblinded_R * k_precisionR) );
    return freq;
  }
  
  // if we've reached this point, we are doing a systematic study, and are looking at a blinded shift
  // relative to our nominal value
  double unblinded_nominal_R = m_nominalR - (m_blindingSign * m_deltaR);
  double unblinded_systematic_R = unblinded_nominal_R + (m_systematicsSign * blindedValue);
  freq = m_reference * ( 1 + (unblinded_systematic_R * k_precisionR) );
  return freq;
}



// ******************************************************************************************************
// ******************************************************************************************************
void Blinders::warning()
{
  std::cout << "                     YYYYYYY       YYYYYYY     OOOOOOOOO     UUUUUUUU     UUUUUUUU" << std::endl;
  std::cout << "                     Y:::::Y       Y:::::Y   OO:::::::::OO   U::::::U     U::::::U" << std::endl;
  std::cout << "                     Y:::::Y       Y:::::Y OO:::::::::::::OO U::::::U     U::::::U" << std::endl;
  std::cout << "                     Y::::::Y     Y::::::YO:::::::OOO:::::::OUU:::::U     U:::::UU" << std::endl;
  std::cout << "                     YYY:::::Y   Y:::::YYYO::::::O   O::::::O U:::::U     U:::::U" << std::endl;
  std::cout << "                        Y:::::Y Y:::::Y   O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                         Y:::::Y:::::Y    O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                          Y:::::::::Y     O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                           Y:::::::Y      O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                            Y:::::Y       O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                            Y:::::Y       O:::::O     O:::::O U:::::D     D:::::U" << std::endl;
  std::cout << "                            Y:::::Y       O::::::O   O::::::O U::::::U   U::::::U" << std::endl;
  std::cout << "                            Y:::::Y       O:::::::OOO:::::::O U:::::::UUU:::::::U" << std::endl;
  std::cout << "                         YYYY:::::YYYY     OO:::::::::::::OO   UU:::::::::::::UU" << std::endl;
  std::cout << "                         Y:::::::::::Y       OO:::::::::OO       UU:::::::::UU" << std::endl;
  std::cout << "                         YYYYYYYYYYYYY         OOOOOOOOO           UUUUUUUUU" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "                             AAA               RRRRRRRRRRRRRRRRR   EEEEEEEEEEEEEEEEEEEEEE" << std::endl;
  std::cout << "                            A:::A              R::::::::::::::::R  E::::::::::::::::::::E" << std::endl;
  std::cout << "                           A:::::A             R::::::RRRRRR:::::R E::::::::::::::::::::E" << std::endl;
  std::cout << "                          A:::::::A            RR:::::R     R:::::REE::::::EEEEEEEEE::::E" << std::endl;
  std::cout << "                         A:::::::::A             R::::R     R:::::R  E:::::E       EEEEEE" << std::endl;
  std::cout << "                        A:::::A:::::A            R::::R     R:::::R  E:::::E" << std::endl;
  std::cout << "                       A:::::A A:::::A           R::::RRRRRR:::::R   E::::::EEEEEEEEEE" << std::endl;
  std::cout << "                      A:::::A   A:::::A          R:::::::::::::RR    E:::::::::::::::E" << std::endl;
  std::cout << "                     A:::::A     A:::::A         R::::RRRRRR:::::R   E:::::::::::::::E" << std::endl;
  std::cout << "                    A:::::AAAAAAAAA:::::A        R::::R     R:::::R  E::::::EEEEEEEEEE" << std::endl;
  std::cout << "                   A:::::::::::::::::::::A       R::::R     R:::::R  E:::::E" << std::endl;
  std::cout << "                  A:::::AAAAAAAAAAAAA:::::A      R::::R     R:::::R  E:::::E       EEEEEE" << std::endl;
  std::cout << "                 A:::::A             A:::::A   RR:::::R     R:::::REE::::::EEEEEEEE:::::E" << std::endl;
  std::cout << "                A:::::A               A:::::A  R::::::R     R:::::RE::::::::::::::::::::E" << std::endl;
  std::cout << "               A:::::A                 A:::::A R::::::R     R:::::RE::::::::::::::::::::E" << std::endl;
  std::cout << "               AAAAAAA                   AAAAAAARRRRRRRR     RRRRRRREEEEEEEEEEEEEEEEEEEEEE" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "                    NNNNNNNN        NNNNNNNN     OOOOOOOOO     TTTTTTTTTTTTTTTTTTTTTTT" << std::endl;
  std::cout << "                    N:::::::N       N::::::N   OO:::::::::OO   T:::::::::::::::::::::T" << std::endl;
  std::cout << "                    N::::::::N      N::::::N OO:::::::::::::OO T:::::::::::::::::::::T" << std::endl;
  std::cout << "                    N:::::::::N     N::::::NO:::::::OOO:::::::OT:::::TT:::::::TT:::::T" << std::endl;
  std::cout << "                    N::::::::::N    N::::::NO::::::O   O::::::OTTTTTT  T:::::T  TTTTTT" << std::endl;
  std::cout << "                    N:::::::::::N   N::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N:::::::N::::N  N::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N N::::N N::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N  N::::N:::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N   N:::::::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N    N::::::::::NO:::::O     O:::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N     N:::::::::NO::::::O   O::::::O        T:::::T" << std::endl;
  std::cout << "                    N::::::N      N::::::::NO:::::::OOO:::::::O      TT:::::::TT" << std::endl;
  std::cout << "                    N::::::N       N:::::::N OO:::::::::::::OO       T:::::::::T" << std::endl;
  std::cout << "                    N::::::N        N::::::N   OO:::::::::OO         T:::::::::T" << std::endl;
  std::cout << "                    NNNNNNNN         NNNNNNN     OOOOOOOOO           TTTTTTTTTTT" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << " BBBBBBBBBBBBBBBBB   LLLLLLLLLLL             IIIIIIIIIINNNNNNNN        NNNNNNNNDDDDDDDDDDDDD" << std::endl;
  std::cout << " B::::::::::::::::B  L:::::::::L             I::::::::IN:::::::N       N::::::ND::::::::::::DDD" << std::endl;
  std::cout << " B::::::BBBBBB:::::B L:::::::::L             I::::::::IN::::::::N      N::::::ND:::::::::::::::DD" << std::endl;
  std::cout << " BB:::::B     B:::::BLL:::::::LL             II::::::IIN:::::::::N     N::::::NDDD:::::DDDDD:::::D" << std::endl;
  std::cout << "   B::::B     B:::::B  L:::::L                 I::::I  N::::::::::N    N::::::N  D:::::D    D:::::D" << std::endl;
  std::cout << "   B::::B     B:::::B  L:::::L                 I::::I  N:::::::::::N   N::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B::::BBBBBB:::::B   L:::::L                 I::::I  N:::::::N::::N  N::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B:::::::::::::BB    L:::::L                 I::::I  N::::::N N::::N N::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B::::BBBBBB:::::B   L:::::L                 I::::I  N::::::N  N::::N:::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B::::B     B:::::B  L:::::L                 I::::I  N::::::N   N:::::::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B::::B     B:::::B  L:::::L                 I::::I  N::::::N    N::::::::::N  D:::::D     D:::::D" << std::endl;
  std::cout << "   B::::B     B:::::B  L:::::L         LLLLLL  I::::I  N::::::N     N:::::::::N  D:::::D    D:::::D" << std::endl;
  std::cout << " BB:::::BBBBBB::::::BLL:::::::LLLLLLLLL:::::LII::::::IIN::::::N      N::::::::NDDD:::::DDDDD:::::D" << std::endl;
  std::cout << " B:::::::::::::::::B L::::::::::::::::::::::LI::::::::IN::::::N       N:::::::ND:::::::::::::::DD" << std::endl;
  std::cout << " B::::::::::::::::B  L::::::::::::::::::::::LI::::::::IN::::::N        N::::::ND::::::::::::DDD" << std::endl;
  std::cout << " BBBBBBBBBBBBBBBBB   LLLLLLLLLLLLLLLLLLLLLLLLIIIIIIIIIINNNNNNNN         NNNNNNNDDDDDDDDDDDDD" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "" << std::endl;
}
                                                                                                   
                                                                                                   
