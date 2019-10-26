#!/usr/bin/env bash

dirty="$(./scripts/format.sh | git diff --no-color)"

failedMsg="C++ formatting test failed, please format your code:"

if [[ $dirty ]]; then
    echo $failedMsg
    echo $dirty
    exit 1
fi

echo "C++ formatting test passed!!"
exit 0