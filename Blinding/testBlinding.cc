#include "Blinders.hh"

using namespace blinding;

int main() {

  Blinders::fitType ftype = Blinders::kOmega_a;
  Blinders myBlinder( ftype );
  
  Blinders getBlinded( ftype, "Chris eats at Two Brothers" );
  
  Blinders systematicallyBlinded( ftype, 1, 10, "Help me! I think I'm falling..." );

  // there should be no blinding...
  std::cout << "\n\n should be unblinded results " << std::endl;
  for ( double R = 0; R < 10; R += 1 ) {
    double result = ( myBlinder.paramToFreq( R ) / myBlinder.referenceValue() ) - 1;
    std::cout << " input R: " << R << "   output: " << result << std::endl;
  }
  // there should be nominal blinding...
  std::cout << "\n\n should be blinded central results " << std::endl;
  for ( double R = 0; R < 10; R += 1 ) {
    double result = ( getBlinded.paramToFreq( R ) / getBlinded.referenceValue() ) - 1;
    std::cout << " input R: " << R << "   output: " << result << std::endl;
  }

  // there should be systematic shifts relative to nominal central...
  std::cout << "\n\n should be systematic shift results " << std::endl;
  for ( double R = 0; R < 10; R += 1 ) {
    double result = ( systematicallyBlinded.paramToFreq( R ) / systematicallyBlinded.referenceValue() ) - 1;
    std::cout << " input R: " << R << "   output: " << result << std::endl;
  }

}
