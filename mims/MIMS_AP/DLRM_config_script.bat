ECHO off
setlocal enableDelayedExpansion
SET LISTM=960 1920 3840 7680 512 1024 2048 4096 8192
SET LISTN=1024 2048 4096 8192 512 1024 2048 4096 8192
SET LISTK=1024 2048 4096 8192 512 1024 2048 4096 8192
SET i=0
for %%b in (%LISTM%) do (
  set /a i+=1
  set ARRAY_M[!i!]=%%b
)
SET i=0
for %%b in (%LISTN%) do (
  set /a i+=1
  set ARRAY_N[!i!]=%%b
)
SET i=0
for %%b in (%LISTK%) do (
  set /a i+=1
  set ARRAY_K[!i!]=%%b
)
rem ECHO on
(for /L %%b in (1,1,%i%) do (
  python mims.py -preset .\presets\1MI100_1gpu_gemm.ini -sw_opt dataformat FP16 -sw_opt noTrail True -gemm_dimms !ARRAY_M[%%b]! !ARRAY_N[%%b]! !ARRAY_K[%%b]! -postfix _!ARRAY_M[%%b]!x!ARRAY_N[%%b]!x!ARRAY_K[%%b]!
))
PAUSE