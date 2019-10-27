#!/bin/bash

### RUN THIS SCRIPT INSIDE YOUR SRC FOLDER ###

# Specify if headers or sources are printed
fileType=$1

# The folder to move from and the folder to move to
inputFolder=$2

if [ $fileType = "-h" ]; then
    readarray -d '' inputHeadersHpp < <(find ${inputFolder} -name "*.hpp" -print0)
    readarray -d '' inputHeadersH < <(find ${inputFolder} -name "*.h" -print0)
    readarray -d '' inputHeadersCuh < <(find ${inputFolder} -name "*.cuh" -print0)
    files=(${inputHeadersHpp[@]} ${inputHeadersH[@]} ${inputHeadersCuh[@]})
fi

if [ $fileType = "-s" ]; then
    readarray -d '' inputSourcesCpp < <(find ${inputFolder} -name "*.cpp" -print0)
    readarray -d '' inputSourcesC < <(find ${inputFolder} -name "*.c" -print0)
    readarray -d '' inputSourcesCu < <(find ${inputFolder} -name "*.cu" -print0)
    files=(${inputSourcesCpp[@]} ${inputSourcesC[@]} ${inputSourcesCu[@]})
fi

# Print in a format easy to copy paste for Qt .pro file
for file in "${files[@]}"
do
    echo "    $file \\"
done