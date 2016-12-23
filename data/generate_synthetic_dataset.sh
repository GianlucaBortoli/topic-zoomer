#!/bin/bash
MIN=$1 # range min
MAX=$2 # range max
TEXT=$3 # where to find text strings, separated by newline
OUT=$4 # the output file

echo "x,y,document" >> "$OUT" # csv header
while IFS='' read -r line || [[ -n "$line" ]]; do
  X=$(shuf -i $MIN-$MAX -n 1)
  Y=$(shuf -i $MIN-$MAX -n 1)
  echo "$X,$Y,$line" >> "$OUT"
done < "$TEXT"
