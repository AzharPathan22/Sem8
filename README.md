# Sem8

# In Ubuntu

## To create folder
```bash
mkdir foldername
```

## To create and edit a file 
```bash
gedit filename.extension
```
- cpp - .cpp
- CUDA - .cu
- python - .py

## Run C++ file (normal)
```bash
g++ filename.cpp -o exefilename
```
```bash
./exefilename
```

## Run C++ file with OpenMP
```bash
g++ -fopenmp filename.cpp -o exefilename
```
```bash
./exefilename
```

## Run CUDA file (Save file with .cu)
```bash
nvcc filename.cu -o exefilename
```
```bash
./exefilename
```
