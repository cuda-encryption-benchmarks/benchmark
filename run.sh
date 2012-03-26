#!/bin/bash
./benchmark serpent serial encrypt input.txt
./benchmark serpent serial decrypt input.txt
./benchmark serpent parallel encrypt small1.txt
./benchmark serpent parallel decrypt small1.txt
./benchmark serpent parallel encrypt small2.txt
./benchmark serpent parallel decrypt small2.txt
