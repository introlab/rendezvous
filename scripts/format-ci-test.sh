#!/usr/bin/env bash

git checkout $TRAVIS_BRANCH
clean="$(git diff)"

./scripts/format.sh  

dirty="$(git diff)"
failedMsg="C++ formatting test failed, please format your code:"

if [ "$dirty" == "$clean" ]; then
    echo "C++ formatting test passed!!"
    exit 0
fi

echo $failedMsg
echo $dirty
exit 1