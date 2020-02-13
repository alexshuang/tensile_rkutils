ECHO off
setlocal enableDelayedExpansion
SET list=16 32 64 96 128 192 256
SET list1=FP16 BF16 FP32
(for %%b in (%list1%) do (
  ECHO cBoxDataFmt=%%b>>"C:\Users\aspanday\Perforce\aspanday_MIMSFirstLook\gfxip\arch\rtg_sd\MIMS_AP\presets\1MI100_1gpu_gemm.ini"
  (for %%a in (%list%) do (
    SET MY_VARIABLE=%%a
    SET /A BS=MY_VARIABLE*15
    ECHO txtABlocks=%%a>>"C:\Users\aspanday\Perforce\aspanday_MIMSFirstLook\gfxip\arch\rtg_sd\MIMS_AP\presets\1MI100_1gpu_gemm.ini"
    ECHO txtBatchSize=!BS!>>"C:\Users\aspanday\Perforce\aspanday_MIMSFirstLook\gfxip\arch\rtg_sd\MIMS_AP\presets\1MI100_1gpu_gemm.ini"
    ECHO txtPostfix=_%%a_128_%%b>>"C:\Users\aspanday\Perforce\aspanday_MIMSFirstLook\gfxip\arch\rtg_sd\MIMS_AP\presets\1MI100_1gpu_gemm.ini"
    python mims.py -preset presets\1MI100_1gpu_gemm.ini
  ))
))
PAUSE