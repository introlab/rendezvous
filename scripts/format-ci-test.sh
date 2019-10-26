#!/usr/bin/env bash

dirty="$(git diff -U0 --no-color HEAD^ | ./scripts/format.sh)"

failedMsg="C++ formatting test failed, please format your code:"

if [[ $dirty ]]; then
    echo $failedMsg
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0