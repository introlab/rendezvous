#!/usr/bin/bash
shader="$1"
inputPath="$2"
outputPath="$3"
output="\""
source="Source"
while read line; do
    if ! [ -z "$line" ]; then
        output="$output$line\n"
    fi    
done < "$inputPath/$shader.txt"
echo "const char * $shader$source = $output\0\";" > $outputPath/$shader.cpp
