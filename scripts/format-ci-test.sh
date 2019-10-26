#!/usr/bin/env bash

chmod +x ./scripts/format.sh
./scripts/format.sh  

dirty=$(git ls-files --modified)
failedMsg="C++ formatting test failed, please format your code:"

if [[ $dirty ]]; then
    echo $failedMsg
    echo $dirty
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0