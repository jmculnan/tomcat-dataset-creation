#!/bin/bash
# use an argument to indicate the path to the files you are converting
# e.g. ./mp4towav.sh this/is/my/wav/path/*.mp4

FILES=$1
for f in $FILES
do
  fname=${f%.*}
  #echo $fname
  newf="$fname.wav"
  #echo $newf
  ffmpeg -i $f $newf
done
