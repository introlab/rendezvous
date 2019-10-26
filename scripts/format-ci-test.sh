#!/usr/bin/env bash

git checkout $TRAVIS_BRANCH
./scripts/format.sh  

dirty="$(git diff)"
failedMsg="C++ formatting test failed, please format your code:"

if [[ $dirty ]]; then
    echo $failedMsg
    echo $dirty
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0