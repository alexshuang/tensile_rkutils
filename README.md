# Tensile Replacement Kernel Development Utils
ROCm tensile replacement kernel development utils.

- **rkgen**&emsp;&emsp;generate replacement kernel template: \*.sp3.
- **sp3cc**&emsp;&emsp;compile \*.sp3 to \*.inc
- **rkbuild**&emsp;&emsp;build \*.co with \*.inc
- **rkperf**&emsp;&emsp;perf the kernel with various kinds of problem sizes

---

## Usage Guide

### Setup
- $ cd tensile_rkutils && source setup_env

### Generate kernel
- $ rkgen -f <*path/to/config file: config/MT64x128x32.json*> [-o *path/to/output.sp3*]

### Compile
- $ sp3cc <*path/to/\*.sp3*>

### Build
- $ cd 1_BenchmarkProblems/\*/00_Final/source/assembly
- $ rkbuild <*path/to/\*.inc*>

### Testing
- $ cd 1_BenchmarkProblems/\*/00_Final/build
- $ rkperf -g conf.json  &emsp; # Create profile
- Modify problem sizes in conf.json
- $ rkperf -f conf.json  &emsp; # Read the output \*.csv using execl.
