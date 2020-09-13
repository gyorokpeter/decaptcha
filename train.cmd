@echo off
echo set to low i/o priority now
pause
PATH=D:\Utility\Python35;%PATH%
rem python -i decaptcha_convnet.py --learning_rate 0.0005 --dropout 0.7
python -i decaptcha_convnet.py