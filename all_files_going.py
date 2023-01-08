import os 
import logging

from pathlib import Path
from track import run


def start_script(k, conf_thres, video_folder_name, txt_folder_name): 

    # Paths to save results: 
    video_name = os.listdir(video_folder_name)
    inp_txt_name = os.listdir(txt_folder_name)

    for num, vid in enumerate(video_name): 
        vid_name = Path(vid).stem 
        txt_name = [txt for txt in inp_txt_name if Path(txt).stem == vid_name] # поиск текстового
            
        if not len(txt_name): 
            logging.warning(f'\nФайл {vid_name}.txt отсутствует в TXT_FOLDER/\
                    \nСтарт обработки видео без GPS-координат.\n')
            command = f'python track.py --source VIDEO_FOLDER/{vid} --yolo-weights weights/best.pt --imgsz 640 --conf-thres {conf_thres} --step_frame {k} --images_fold RESULT/{vid_name}/IMAGES/ --vid_name {vid_name}'
            res = os.system(command)
        else:
            # run(
            #     source=Path(f'VIDEO_FOLDER/{vid}'), 
            #     yolo_weights=Path('weights/best.pt'),
            #     imgsz=(640, 640), 
            #     conf_thres=conf_thres,
            #     step_frame=k,
            #     images_fold=Path(f'RESULT/IMAGES/{vid_name}/'), 
            #     gps_txt_source=Path(f'TXT_FOLDER/{txt_name[0]}')
            # )
            command = f'python track.py --source VIDEO_FOLDER/{vid} --yolo-weights weights/best.pt --gps-txt-source TXT_FOLDER/{txt_name[0]} --imgsz 640 --conf-thres {conf_thres} --step_frame {k} --images_fold RESULT/{vid_name}/IMAGES/ --vid_name {vid_name}'
            res = os.system(command)
            

