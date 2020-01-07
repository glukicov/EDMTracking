# EDMTracking
g-2 EDM, grid-based tracking, and DB scripts 

### To get the EDM-style blinding to work in Python with the blinding libraries: ### 
A. Add changes to your local files from this commit: https://github.com/glukicov/EDMTracking/commit/e2c0979d45648bd0d9f09aa5de76d6b75dc4a183 

B. Re-compile (in the Blinders.cc directroy):
```
g++ -I rlib/include -I /usr/local/opt/openssl/include Blinders.cc -std=c++11 -Wall -Wextra -Werror -pedantic-errors -fpic -c
```
C. Now re-create a shared object library that can be used with both ROOT and Python:
```
g++ -shared -o libBlinders.so Blinders.o rlib/src/Random.o -L/usr/local/opt/openssl/lib -lssl -lcrypto
```
D. Now construct your blinder *with 5 input arguments* as follows (e.g.):
```
getBlinded = Blinders(FitType.Omega_a, blinding_string, boxWidth, gausWidth, "edm")
```
The final 'edm' string argument is just a dummy parameter (can be anything!) to distinguish between the other systematic constructor

The official blinding guide is here: https://cdcvs.fnal.gov/redmine/projects/gm2analyses/wiki/Library_installation 

The getting-started Python guide is here: https://cdcvs.fnal.gov/redmine/projects/gm2analyses/wiki/Python-based_EDM_analysis

Example of EDM Tracking art (C++) code is here https://cdcvs.fnal.gov/redmine/projects/gm2analyses/wiki/Tracker_EDM_analysis
