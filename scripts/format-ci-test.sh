#!/usr/bin/env bash

clean=$(git ls-files --modified)
chmod +x ./scripts/format.sh
./scripts/format.sh  

dirty=$(git ls-files --modified)
failedMsg="C++ formatting test failed, please format your code:"

if [[ "$dirty" == "$clean" ]]; then
    echo "C++ formatting test passed!!"
    exit 0
fi

echo $failedMsg
echo "$dirty"
exit 1