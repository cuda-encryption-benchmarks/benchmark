#!/bin/bash

# Serial
echo "Serial"
sha256sum input.txt small1.txt small2.txt
./benchmark serpent serial encrypt input.txt
./benchmark serpent serial decrypt input.txt
./benchmark serpent serial encrypt small1.txt
./benchmark serpent serial decrypt small1.txt
./benchmark serpent serial encrypt small2.txt
./benchmark serpent serial decrypt small2.txt
sha256sum input.txt small1.txt small2.txt

# Parallel
echo "Parallel"
sha256sum input.txt small1.txt small2.txt
./benchmark serpent parallel encrypt input.txt
./benchmark serpent parallel decrypt input.txt
./benchmark serpent parallel encrypt small1.txt
./benchmark serpent parallel decrypt small1.txt
./benchmark serpent parallel encrypt small2.txt
./benchmark serpent parallel decrypt small2.txt
sha256sum input.txt small1.txt small2.txt

# CUDA
echo "CUDA"
sha256sum input.txt small1.txt small2.txt
./benchmark serpent cuda encrypt input.txt
./benchmark serpent cuda decrypt input.txt
./benchmark serpent cuda encrypt small1.txt
./benchmark serpent cuda decrypt small1.txt
./benchmark serpent cuda encrypt small2.txt
./benchmark serpent cuda decrypt small2.txt
sha256sum input.txt small1.txt small2.txt
