#!/bin/bash

### RUN THIS SCRIPT INSIDE YOUR SRC FOLDER ###

# The folder to move from and the folder to move to
inputFolder=$1
outputFolder=$2

# Get all files in input folder
readarray -d '' inputHeadersHpp < <(find ${inputFolder} -name "*.hpp" -print0)
readarray -d '' inputHeadersH < <(find ${inputFolder} -name "*.h" -print0)
readarray -d '' inputHeadersCuh < <(find ${inputFolder} -name "*.cuh" -print0)
readarray -d '' inputSourcesCpp < <(find ${inputFolder} -name "*.cpp" -print0)
readarray -d '' inputSourcesC < <(find ${inputFolder} -name "*.c" -print0)
readarray -d '' inputSourcesCu < <(find ${inputFolder} -name "*.cu" -print0)
inputHeaders=(${inputHeadersHpp[@]} ${inputHeadersH[@]} ${inputHeadersCuh[@]})
inputSources=(${inputSourcesCpp[@]} ${inputSourcesC[@]} ${inputSourcesCu[@]})

# Format header files to remove ./
for ((i=0; i<${#inputHeaders[@]}; i++));
do
    inputHeaders[$i]=${inputHeaders[$i]#"./"}
done

# Copy the inputHeaders and modify path
newPathHeaders=("${inputHeaders[@]}")

for ((i=0; i<${#newPathHeaders[@]}; i++));
do
    newPathHeaders[$i]="${outputFolder}/${newPathHeaders[$i]#"$inputFolder/"}"
    mkdir -p ${newPathHeaders[$i]%/*}
done

# Move all header files in input folder
for ((i=0; i<${#inputHeaders[@]}; i++));
do
    if [ "${inputHeaders[$i]}" != "${newPathHeaders[$i]}" ]; then
        mv ${inputHeaders[$i]} ${newPathHeaders[$i]}
    fi
done

# Move all source files in input folder
for source in "${inputSources[@]}"
do
    newPathSource="${outputFolder}/${source#"$inputFolder/"}"

    if [ "$source" != "$newPathSource" ]; then
        mv $source $newPathSource
    fi
done

# Remove the empty folder (only if ouput is not a sub folder of input)
if [[ $outputFolder != *$inputFolder* ]]; then
    rm -r $inputFolder
else
    # Remove empty subfolders of source files
    for source in "${inputSources[@]}"
    do
        sourceSubFolder=${source%/*}

        if [[ -e $sourceSubFolder ]] && [[ $outputFolder != *$sourceSubFolder* ]]; then
            rm -r $sourceSubFolder
        fi
    done

    # Remove empty subfolders of header files
    for header in "${inputHeaders[@]}"
    do
        headerSubFolder=${header%/*}

        if [[ -e $headerSubFolder ]] && [[ $outputFolder != *$headerSubFolder* ]]; then
            rm -r $headerSubFolder
        fi
    done
fi

# Get all files
readarray -d '' headersHpp < <(find . -name "*.hpp" -print0)
readarray -d '' headersH < <(find . -name "*.h" -print0)
readarray -d '' headersCuh < <(find . -name "*.cuh" -print0)
readarray -d '' sourcesCpp < <(find . -name "*.cpp" -print0)
readarray -d '' sourcesC < <(find . -name "*.c" -print0)
readarray -d '' sourcesCu < <(find . -name "*.cu" -print0)
headers=(${headersHpp[@]} ${headersH[@]} ${headersCuh[@]})
sources=(${sourcesCpp[@]} ${sourcesC[@]} ${sourcesCu[@]})

# Modify the includes in all header files
for header in "${headers[@]}"
do
    # Change all our includes in the header file to new path
    for ((i=0; i<${#inputHeaders[@]}; i++));
    do
        sed -i "s@${inputHeaders[$i]}@${newPathHeaders[$i]}@g" $header
    done
done

# Modify the includes in all source files
for source in "${sources[@]}"
do
    # Change all our includes in the source file to new path
    for ((i=0; i<${#inputHeaders[@]}; i++));
    do
        sed -i "s@${inputHeaders[$i]}@${newPathHeaders[$i]}@g" $source
    done
done