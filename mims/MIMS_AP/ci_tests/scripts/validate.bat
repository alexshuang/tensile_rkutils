@set /A ERR_COUNT=0
@cd ../..

python mims.py -preset mi100_1gpu_deepspeech2.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi100_1gpu_resenet50.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi100_8gpu_2box_deepspeech2.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi100_8gpu_2box_resnet50.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi100_8gpu_2box_transformer.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi100_8gpu_fb.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%
python mims.py -preset mi200_8gpu_fb.ini -validate
@set /A ERR_COUNT=%ERR_COUNT%+%ERRORLEVEL%

@cd ci_tests/scripts
@IF /I "%ERR_COUNT%" GEQ "1" set /A ERR_COUNT=1
@exit /b %ERR_COUNT%