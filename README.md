## Introduction
<p align="justify">
This repository contains a speedup implementation in GPU of Swith Waterman algorithm, written in CUDA language.
This project has been developed during the course of GPU101 by Politecnico di Milano and the original algorithm came from Alberto Zeni.
</p>

## Usage

### Compilation

To compile basic version of the algorithm, written in C, you need only GCC compiler.
Since speedup implementation of this code is written in CUDA, you also need NVCC compiler, CUDA package and an NVIDIA GPU.
The command for compiling is simply:

nvcc 'program.cu' -o 'program'

No other parameters are requested, since the query and reference are generated randomly during the execution!

