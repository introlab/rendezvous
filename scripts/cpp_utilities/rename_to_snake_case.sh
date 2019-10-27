#!/bin/bash

### RUN THIS SCRIPT INSIDE YOUR SRC FOLDER ###

# Get all files
readarray -d '' headersHpp < <(find . -name "*.hpp" -print0)
readarray -d '' headersH < <(find . -name "*.h" -print0)
readarray -d '' headersCuh < <(find . -name "*.cuh" -print0)
readarray -d '' sourcesCpp < <(find . -name "*.cpp" -print0)
readarray -d '' sourcesC < <(find . -name "*.c" -print0)
readarray -d '' sourcesCu < <(find . -name "*.cu" -print0)
headers=(${headersHpp[@]} ${headersH[@]} ${headersCuh[@]})
sources=(${sourcesCpp[@]} ${sourcesC[@]} ${sourcesCu[@]})

# Format header files to remove ./
for ((i=0; i<${#headers[@]}; i++));
do
    headers[$i]=${headers[$i]#"./"}
done

# Copy the headers and convert to snake case
snakeCaseHeaders=("${headers[@]}")

for ((i=0; i<${#snakeCaseHeaders[@]}; i++));
do
    snakeCaseHeaders[$i]=$(echo ${snakeCaseHeaders[$i]} | sed 's@\([A-Z]\)@_\L\1@g;s@/_@/@;s@^_@@')
done

# Process header files
for ((i=0; i<${#headers[@]}; i++));
do
    # Rename header file to snake case
    if [ "${headers[$i]}" != "${snakeCaseHeaders[$i]}" ]; then
        mv ${headers[$i]} ${snakeCaseHeaders[$i]}
    fi

    # Change all our includes in the header file to snake case
    for ((j=0; j<${#headers[@]}; j++));
    do
        sed -i "s@${headers[$j]}@${snakeCaseHeaders[$j]}@g" ${snakeCaseHeaders[$i]}
    done
done

# Process all source files
for source in "${sources[@]}"
do
    snakeCaseSource=$(echo $source | sed 's@\([A-Z]\)@_\L\1@g;s@/_@/@;s@^_@@')

    # Rename source file to snake case
    if [ "$source" != "$snakeCaseSource" ]; then
        mv $source $snakeCaseSource
    fi

    # Change all our includes in the source file to snake case
    for ((j=0; j<${#headers[@]}; j++));
    do
        sed -i "s@${headers[$j]}@${snakeCaseHeaders[$j]}@g" $snakeCaseSource
    done

    # Change includes which doesn't have full path
    for ((j=0; j<${#headers[@]}; j++));
    do
        sed -i "s@\"${headers[$j]##*/}\"@\"${snakeCaseHeaders[$j]##*/}\"@g" $snakeCaseSource
    done
done