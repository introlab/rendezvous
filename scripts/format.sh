#!/usr/bin/env bash

find ./steno/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \; 
