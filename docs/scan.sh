#### script to convert four ($1-$4) images into one ($5) row by row 
## and display in the terminal using imgcat 

# usage convert.sh {$1} {$2} {$3} {$4} {$5}
# e.g ../docs/convert.sh fft_S18_60h.png fft_S18_EG.png fft_S18_9D.png fft_S18_HK.png FFT_S18.png

convert +append theta_A_Bz_S12_60h.png theta_A_Bz_S18_60h.png bz.png # row (1, 2) 

convert +append theta_chi2_S12_60h.png theta_chi2_S18_60h.png chi2.png # row (1, 2) 

convert -append bz.png chi2.png scan.png