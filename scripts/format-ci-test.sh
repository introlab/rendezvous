#!/usr/bin/env bash

dirty="$(find ./c_plus_plus/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} -output-replacements-xml \;)"

failedMsg="C++ formatting test failed, please format your code:"

if [[ $dirty ]]; then
    echo $failedMsg
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0