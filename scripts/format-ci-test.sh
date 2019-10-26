#!/usr/bin/env bash

find ./c_plus_plus/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \; 

failedMsg="C++ formatting test failed because of these files:"

git checkout $TRAVIS_BRANCH
dirty=$(git ls-files --modified)

if [[ $dirty ]]; then
    echo $failedMsg
    echo $dirty
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0