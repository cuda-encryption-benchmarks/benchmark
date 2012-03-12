#!/bin/bash
echo "Input data here." > input.txt
./benchmark serpent cuda encrypt input.txt
echo "1:"
./benchmark serpent cuda encrypt input.txt
echo "2:"
./benchmark serpent cuda encrypt input.txt
echo "3:"
./benchmark serpent cuda encrypt input.txt
echo "4:"
./benchmark serpent cuda encrypt input.txt
echo "5:"
