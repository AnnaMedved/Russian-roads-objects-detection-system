import os 
from all_files_going import start_script


# Checking & install all requirements: 
upd_pip = 'pip install --update-pip'
upd = os.system(upd_pip)
instll_all_libs = 'pip install -r requirements.txt'
instll = os.system(instll_all_libs)

# ДЛЯ ОБРАБОТКИ КАЖДОГО K-ОГО КАДРА: 
k = 1
# ПОРОГ ВЕРОЯТНОСТИ (Разделитель - точка): 
conf_thres = 0.3 

start_script(k, conf_thres, 'VIDEO_FOLDER', 'TXT_FOLDER')
