# Parallel-Processing-cpp-Openmp-MPI-Cuda

This repository contains various implementations of numerical and image processing algorithms using different parallel computing frameworks in c/c++.

## 1. Affine Transformation (C)
- Implements 2D affine transformations on images (rotation, translation, scaling, etc.).
- Written in standard C.

## 2. CUDA – Gauss Elimination Method
- Solves systems of linear equations using Gaussian elimination.
- Parallelized using NVIDIA CUDA for GPU acceleration.

## 3. MPI – Convolution & MaxPooling
- Distributed image convolution and max pooling using MPI.
- Suitable for multi-node environments or high-performance computing clusters.

## 4. OpenMP – Convolution & MaxPooling
- Applies convolution and max pooling on images using OpenMP.
- Exploits multi-threading on CPU cores for performance.

## 5. OpenMP – Gauss Elimination Method
- CPU-based parallel solution of linear systems using OpenMP.
- Compares favorably to serial implementations.
