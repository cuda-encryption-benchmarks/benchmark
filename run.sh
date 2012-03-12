#!/bin/bash
#echo "Input data here." > input.txt
./benchmark serpent cuda encrypt input.txt
./benchmark serpent serial decrypt input.txt
