#!/bin/bash

input=$1
inputFolder=$2

if ! [ -e $input ]; then
    echo "Error : Input file does not exist!"
elif [ -z "$inputFolder" ]; then
    echo "Error : No input folder was specified!"
elif [[ $input != *".pro" ]]; then
    echo "Error : Input file is not a qt .pro file!"
else
    readarray -d '' inputHeadersHpp < <(find ${inputFolder} -name "*.hpp" -print0 | sort -z)
    readarray -d '' inputHeadersH < <(find ${inputFolder} -name "*.h" -print0 | sort -z)
    readarray -d '' inputHeadersCuh < <(find ${inputFolder} -name "*.cuh" -print0 | sort -z)
    readarray -d '' inputSourcesCpp < <(find ${inputFolder} -name "*.cpp" -print0 | sort -z)
    readarray -d '' inputSourcesC < <(find ${inputFolder} -name "*.c" -print0 | sort -z)
    readarray -d '' inputSourcesCu < <(find ${inputFolder} -name "*.cu" -print0 | sort -z)
    inputHeaders=(${inputHeadersHpp[@]} ${inputHeadersH[@]} ${inputHeadersCuh[@]})
    inputSources=(${inputSourcesCpp[@]} ${inputSourcesC[@]})

    skipLines=0
    output=()

    # Read input .pro file and write it's modified content to output array
    while IFS= read -r line
    do
        # Skip lines which contains file we want to override
        if [ $skipLines == 1 ]; then
            if [[ $line != *"\\"* ]]; then
                skipLines=0
            fi
        else
            output+=("$line")
            
            # When the sources are reached, skip lines and output new sources
            if [[ $line == "SOURCES"* ]]; then
                skipLines=1

                size=${#inputSources[@]}
                lastIndex=$(($size - 1))

                for ((i=0; i<$size; i++));
                do
                    if (( $i < $lastIndex )); then
                        output+=("    ${inputSources[$i]} \\")
                    else
                        output+=("    ${inputSources[$i]}")
                    fi
                done
            fi

            # When the headers are reached, skip lines and output new headers
            if [[ $line == "HEADERS"* ]]; then
                skipLines=1

                size=${#inputHeaders[@]}
                lastIndex=$(($size - 1))

                for ((i=0; i<$size; i++));
                do
                    if (( $i < $lastIndex )); then
                        output+=("    ${inputHeaders[$i]} \\")
                    else
                        output+=("    ${inputHeaders[$i]}")
                    fi
                done
            fi

            # When the cuda sources are reached, skip lines and output new cuda sources
            if [[ $line == "CUDA_SOURCES"* ]]; then
                skipLines=1

                size=${#inputSourcesCu[@]}
                lastIndex=$(($size - 1))

                for ((i=0; i<$size; i++));
                do
                    if (( $i < $lastIndex )); then
                        output+=("    ${inputSourcesCu[$i]} \\")
                    else
                        output+=("    ${inputSourcesCu[$i]}")
                    fi
                done
            fi
        fi
    done < "$input"

    # Clean input .pro file
    > $input

    # Write new content to input .pro file
    for line in "${output[@]}"
    do
        echo "$line" >> $input
    done
fi