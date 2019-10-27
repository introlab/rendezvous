#!/bin/bash

inputFolder=$1

readarray -d '' inputHeadersHpp < <(find ${inputFolder} -name "*.hpp" -print0 | sort -z)
readarray -d '' inputHeadersH < <(find ${inputFolder} -name "*.h" -print0 | sort -z)
readarray -d '' inputHeadersCuh < <(find ${inputFolder} -name "*.cuh" -print0 | sort -z)
readarray -d '' inputSourcesCpp < <(find ${inputFolder} -name "*.cpp" -print0 | sort -z)
readarray -d '' inputSourcesC < <(find ${inputFolder} -name "*.c" -print0 | sort -z)
readarray -d '' inputSourcesCu < <(find ${inputFolder} -name "*.cu" -print0 | sort -z)
inputHeaders=(${inputHeadersHpp[@]} ${inputHeadersH[@]} ${inputHeadersCuh[@]})
inputSources=(${inputSourcesCpp[@]} ${inputSourcesC[@]})

for inputHeader in "${inputHeaders[@]}"
do
    echo "" >> $inputHeader
done

for inputSource in "${inputSources[@]}"
do
    echo "" >> $inputSource
done
