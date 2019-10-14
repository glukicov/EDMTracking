from ctypes import cdll
import ctypes
lib = cdll.LoadLibrary('./libBlinders.so')

class FitType:
    Omega_a, Omega_p = range(0, 2)

class Blinders(object):
    def __init__(self, fit_type, *args):
        numArgs = len(args)
        if (numArgs == 0):
            lib.Blinders_unblinded.restype = ctypes.c_void_p
            lib.Blinders_unblinded.argtypes = [ctypes.c_int]
            self.obj = lib.Blinders_unblinded(fit_type)
        elif (numArgs == 1):
            blindingString = args[0].encode()
            lib.Blinders_blinded.restype = ctypes.c_void_p
            lib.Blinders_blinded.argtypes = [ctypes.c_int, ctypes.c_char_p]
            self.obj = lib.Blinders_blinded(fit_type,blindingString)
        elif (numArgs == 3):
            studyIndex = int(args[0])
            nominalR = float(args[1]) # assuming value has no more than 15 dp
            blindingString = args[2].encode()
            bS = blindingString.replace('"'.encode(),'\\"'.encode())

            lib.Blinders_sys_blinded.restype = ctypes.c_void_p
            lib.Blinders_sys_blinded.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_char_p]
            self.obj = lib.Blinders_sys_blinded(fit_type,studyIndex,ctypes.c_double(nominalR),bS)
        else:
            print('UNKNOWN CONSTRUCTOR')
    def paramToFreq(self,R):
        lib.Blinders_paramToFreq.argtypes = [ctypes.c_void_p,ctypes.c_double]
        lib.Blinders_paramToFreq.restype = ctypes.c_double
        return lib.Blinders_paramToFreq(self.obj,ctypes.c_double(R))
    def referenceValue(self):
        lib.Blinders_referenceValue.argtypes = [ctypes.c_void_p]
        lib.Blinders_referenceValue.restype = ctypes.c_double
        return lib.Blinders_referenceValue(self.obj)

