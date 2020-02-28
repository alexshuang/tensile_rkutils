#!/bin/sh

set -e

#RUNDIR=~/tony/MT64x128x32/1_BenchmarkProblems/Cijk_Alik_Bljk_HBH_00/00_Final/source/assembly
RUNDIR=bf16_out/1_BenchmarkProblems/Cijk_Alik_Bljk_BBH_00/00_Final/source/assembly/

rkgen -f config/bf16_MT64x128x32.json
cp bf16_MT64x128x32.sp3 ${RUNDIR} -v
cd ${RUNDIR}
sp3cc bf16_MT64x128x32.sp3 && rkbuild bf16_MT64x128x32.inc
