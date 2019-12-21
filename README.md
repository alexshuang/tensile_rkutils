# Tensile Replacement Kernel Development Utils
utils support mi100 replacement kernel develop & perf, including compiler collection, build and perf.

- *sp3cc*&emsp;compile \*.sp3 to \*.inc
- *rkbuild*&emsp;build \*.co with \*.inc
- *rkperf*&emsp;perf the kernel with various kinds of problem sizes

---

## Simple Guide

### Setup
- $ cd tensile_rkutils && source setup_env

### Compile
- $ sp3cc \*.sp3

### Build
- $ cd 1_BenchmarkProblems/\*/00_Final/source/assembly
- $ rkbuild \*.inc

### Perf
- $ cd 1_BenchmarkProblems/\*/00_Final/build
- $ rkperf -g conf.json
- modify the problem size in the conf.json if you want
- $ rkperf -f conf.json
- *note*: the output--\*.csv can read by execl app.
