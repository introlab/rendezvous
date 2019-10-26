#!/usr/bin/env bash

clean="$(git diff)"

./scripts/format.sh  

dirty="$(git diff)"
failedMsg="C++ formatting test failed, please format your code:"

if [ $dirty != $clean ]; then
    echo $failedMsg
    echo $dirty
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0