#!/usr/bin/env bash

find ./c_plus_plus/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \; 