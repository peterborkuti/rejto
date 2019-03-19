#!/usr/bin/env bash

WIDTH=1000
rm rejto.txt

for i in htm/*.htm; do
    html2text -width $WIDTH -o tmp.txt $i;
    sed -e 's/^ *//;s/ *$//; s/=\+//' tmp.txt|grep -v '^$'>> rejto.txt
done;

#rm tmp.txt
