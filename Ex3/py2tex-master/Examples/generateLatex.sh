#!/bin/bash

dir="python"
out="output.txt"
if [ ! -d "txt/" ] 
then
    mkdir "txt/"
fi
if [ ! -d "pdf/" ] 
then
    mkdir "pdf/"
fi
if [ ! -d "png/" ] 
then
    mkdir "png/"
fi
for wholename in $dir/*.py;
do
  python ../py2tex.py $wholename 1>$out
  filename=${wholename##*/}
  rootname="${filename%.*}"
  #echo $wholename
  #echo $filename
  #echo $rootname
  cp $out "txt/$rootname.txt"
  (pdflatex main.tex -pdf) 
  #>/dev/null
  pdftoppm main.pdf "png/$rootname" -png
  mv main.pdf "pdf/$rootname.pdf"
  
done
#rm tmp.fls
rm main.log
rm main.aux
#rm tmp.fdb_latexmk
rm output.txt

