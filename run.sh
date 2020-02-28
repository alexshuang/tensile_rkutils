#!/bin/sh

set -e

RUNDIR=~/tony/MT64x128x32/1_BenchmarkProblems/Cijk_Alik_Bljk_HBH_00/00_Final/source/assembly

rkgen -f config/MT64x128x32.json
cp MT64x128x32.sp3 ${RUNDIR} -v
cd ${RUNDIR}
sp3cc MT64x128x32.sp3 && rkbuild MT64x128x32.inc
