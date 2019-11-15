#!/usr/bin/env bash

find ./steno/src/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \; 
