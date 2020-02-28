#### script to convert four ($1-$4) images into one ($5) row by row 
## and display in the terminal using imgcat 

# usage convert.sh {$1} {$2} {$3} {$4} {$5}
# e.g ../docs/convert.sh fft_S18_60h.png fft_S18_EG.png fft_S18_9D.png fft_S18_HK.png FFT_S18.png

convert +append {$1} {$2} C1.png # row (1, 2) 
convert +append {$3} {$4} C2.png # row (3, 4)
convert -append C1.png C2.png $5 # stuck columns
/Users/gleb/.iterm2/imgcat $5
