#### script to convert four ($1-$4) images into one ($5) row by row 
## and display in the terminal using imgcat 

# usage convert.sh {$1} {$2} {$3} {$4} {$5}
# e.g ../docs/convert.sh fft_S18_60h.png fft_S18_EG.png fft_S18_9D.png fft_S18_HK.png FFT_S18.png

convert +append  count_60h_S12.png bz_60h_S12.png 60h_S12.png # row (1, 2) 
convert +append  count_9D_S12.png  bz_9D_S12.png 9D_S12.png # row (3, 4)
convert +append  count_HK_S12.png  bz_HK_S12.png HK_S12.png # row (3, 4)
convert +append  count_EG_S12.png  bz_EG_S12.png EG_S12.png # row (3, 4)
convert -append 60h_S12.png 9D_S12.png HK_S12.png  EG_S12.png S12.png # stuck columns

convert +append count_60h_S18.png bz_60h_S18.png 60h_S18.png # row (1, 2) 
convert +append count_9D_S18.png  bz_9D_S18.png 9D_S18.png # row (3, 4)
convert +append count_HK_S18.png  bz_HK_S18.png HK_S18.png # row (3, 4)
convert +append count_EG_S18.png  bz_EG_S18.png EG_S18.png # row (3, 4)
convert -append 60h_S18.png 9D_S18.png HK_S18.png  EG_S18.png S18.png # stuck columns

convert append S12.png S18.png EDM.png