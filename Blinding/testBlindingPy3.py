from BlindersPy3 import Blinders
from BlindersPy3 import FitType

# unblinded fit
myBlinder = Blinders(FitType.Omega_a) # or Blinder(FitType.Omega_p)

# blinded instance
getBlinded = Blinders(FitType.Omega_a, "Chris eats at Two Brothers")

# blinded instance for systematics
systematicallyBlinded = Blinders(FitType.Omega_a, 1, 10, "Help me! I think I'm falling..." )

print('\n\n Unblinded results')
R = 0.0
while (R < 10.0):
    result = ( myBlinder.paramToFreq(R) / myBlinder.referenceValue() ) - 1
    print(' input R: {0:.0f} output: {1:.5e}'.format(R,result))
    R = R + 1

R = 0.0
print('\n\n Blinded central results')
while (R < 10.0):
    result = ( getBlinded.paramToFreq(R) / getBlinded.referenceValue() ) - 1
    print(' input R: {0:.0f} output: {1:.5e}'.format(R,result))
    R = R + 1

print('\n\n Systematic shift results')
R = 0.0
while (R < 10.0):
    result = ( systematicallyBlinded.paramToFreq(R) / systematicallyBlinded.referenceValue() ) - 1;
    print(' input R: {0:.0f} output: {1:.5e}'.format(R,result))
    R = R + 1
