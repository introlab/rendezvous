#!/bin/bash

namespace=$1
inputFolder=$2

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
    output=()
    namespaceStart=0
    namespaceEnd=0
    
    # Parse header file
    while IFS= read -r line
    do
        # Make sure we only add namespace once
        if [ $namespaceStart == 0 ]; then
            # Namespace declaration will go in front of first class/struct/namespace declaration
            if [[ $line == "class"* ]] || [[ $line == "struct"* ]] || [[ $line == "namespace"* ]] || [[ $line == "template"* ]] || [[ $line == "enum"* ]] || [[ $line == *";" ]]; then
                output+=("namespace $namespace")
                output+=("{")
                output+=("")
                namespaceStart=1
            fi
        fi
        
        # Make sure we only close namespace once
        if [ $namespaceStart == 1 ] && [ $namespaceEnd == 0 ]; then
            # Namespace end will be at end of the header file
            if [[ $line == "#endif"* ]]; then
                output+=("} // $namespace")
                output+=("")
                namespaceEnd=1
            fi
        fi

        output+=("$line")

    done < "$inputHeader"

    # Clean header file
    > $inputHeader

    # Write new content to header file
    for line in "${output[@]}"
    do
        echo "$line" >> $inputHeader
    done
done

for inputSource in "${inputSources[@]}"
do
    output=()
    namespaceStart=0
    
    # Parse header file
    while IFS= read -r line
    do
        # Make sure we only add namespace once
        if [ $namespaceStart == 0 ]; then
            # Namespace declaration will go in front of first class/struct/namespace declaration
            if [[ $line == *"::"* ]] || [[ $line == "namespace"* ]]; then
                output+=("namespace $namespace")
                output+=("{")
                output+=("")
                namespaceStart=1
            fi
        fi

        output+=("$line")

    done < "$inputSource"

    output+=("} // $namespace")

    # Clean header file
    > $inputSource

    # Write new content to header file
    for line in "${output[@]}"
    do
        echo "$line" >> $inputSource
    done
done
