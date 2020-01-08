Blinders* Blinders_unblinded(blinding::Blinders::fitType fit_type)     { return new Blinders(fit_type); }
Blinders* Blinders_blinded(blinding::Blinders::fitType fit_type, char* blindingString)     { return new Blinders(fit_type,blindingString); }
Blinders* Blinders_sys_blinded(blinding::Blinders::fitType fit_type, int studyIndex, double nominalR, char* blindingString)   { return new Blinders(fit_type, studyIndex, nominalR, blindingString); }

double Blinders_paramToFreq(Blinders* b, double R){ return b->paramToFreq(R); }
double Blinders_referenceValue(Blinders* b){ return b->referenceValue(); }
