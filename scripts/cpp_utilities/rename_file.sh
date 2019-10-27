#!/bin/bash

file=$1
newName=$2

# Get all files
readarray -d '' headersHpp < <(find . -name "*.hpp" -print0)
readarray -d '' headersH < <(find . -name "*.h" -print0)
readarray -d '' headersCuh < <(find . -name "*.cuh" -print0)
readarray -d '' sourcesCpp < <(find . -name "*.cpp" -print0)
readarray -d '' sourcesC < <(find . -name "*.c" -print0)
readarray -d '' sourcesCu < <(find . -name "*.cu" -print0)
headers=(${headersHpp[@]} ${headersH[@]} ${headersCuh[@]})
sources=(${sourcesCpp[@]} ${sourcesC[@]} ${sourcesCu[@]})

# Change file name
mv $file $newName

# Change include name in headers
for header in "${headers[@]}"
do
    sed -i "s@$file@$newName@g" $header
done

# Change include name in sources
for source in "${sources[@]}"
do
    sed -i "s@$file@$newName@g" $source
done