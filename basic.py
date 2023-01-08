from all_files_going import start_script


# ДЛЯ ОБРАБОТКИ КАЖДОГО K-ОГО КАДРА: 
k = 1
# ПОРОГ ВЕРОЯТНОСТИ (Разделитель - точка): 
conf_thres = 0.3 

start_script(k, conf_thres, 'VIDEO_FOLDER', 'TXT_FOLDER')
