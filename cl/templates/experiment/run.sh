#!/bin/bash
current_dir=$(basename "$PWD")
cd ../../
python3 -m experiment."$current_dir".expRun
