import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import pandas as pd 
import json 
import uuid 

from datetime import datetime, timedelta 
from pathlib import Path
from utils.plots import plot_images


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

classes_names = [
    'snowdrift', 'snow', 'ice', 'snow_track', 'snow_mud', 'snowed_sign', 'melted_snow'
]

translates = [
    'Снежный вал', 'Рыхлый снег', 'Стекловидный лед', 'Снежный накат', 'Грязь', 'Заснеженный знак', 'Зимняя скользкость'
    ]
        
labls = {num: name for num, name in enumerate(classes_names)}

change_value_labels = {classes_names[i]: translates[i] for i in range(len(classes_names))}


@torch.no_grad()
def run(
        source='0',
        gps_txt_source = '', # path to GPSpack,
        images_fold='IMAGES/',
        vid_name='',
        step_frame = 1, # Working with each Ks frame 
        yolo_weights=WEIGHTS / 'best_said.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(1280, 1280),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    save_vid = True
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
            
    if images_fold:
        folder = images_fold
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if step_frame: 
        k = step_frame 

    if gps_txt_source: 
        # Определение fps входного видео: 
        vid = cv2.VideoCapture(source)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        LOGGER.info('\nInput video fps: {}\n'.format(fps))

        # Разрешение: 
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        mass_h = range(0, int(height/2)+1, int(height/8)) # 4 уровня 
        intervs_h = [(mass_h[i-1], (mass_h[i-1]+mass_h[i])/2, mass_h[i]) for i, obj in enumerate(mass_h) if i!= 0]
        coeffs_range = [1, 5, 10, 15, 20] # [1, ... 20] for lower to upper 

        # Интервалы высоты в пикселях: соотв. коэффициенты масштабирования
        coeffs = {
            interv: [coeffs_range[i], coeffs_range[i+1]] 
            for i, interv in enumerate(intervs_h) if i!=len(intervs_h)
            }

        # Определение количества записей во входном файле на 1 сек: 
        with open(gps_txt_source, 'r') as f: 
            inst_per_second = 0
            format = '%d.%m.%Y %H:%M:%S'
            prev_line = f.readline()
            prev_line = f.readline()
            prev_time = datetime.strptime(prev_line[:18], format)
            changed = False 
            txt_lines = f.readlines()
            for line in txt_lines[2:60]: 
                time = datetime.strptime(line[:18], format)
                delta_time = time - prev_time 
                if (changed==False) & (delta_time!=timedelta(0)): 
                    changed = True
                elif (changed==True) & (delta_time!=timedelta(0)): 
                    break
                if changed == True: 
                    inst_per_second += 1
                prev_time = time 

    bboxes_list = []

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
     
    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    # variables for num of objects: 
    bb_all = 0
    im_bb = 0

    # ДЕТЕКЦИЯ ДЛЯ КАЖДОГО K-ОГО кадра: 
    k = step_frame 

    LOGGER.info(f"\nСтарт обработки видео для каждого {k}-ого кадра:\n")
    

    t1_all, t2_all = 0, 0

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if (frame_idx % k) == 0:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                seen += 1
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

                curr_frames[i] = im0

                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop

                #annotator = Annotator(im0, line_width=2, pil=not ascii)
                annotator = Annotator(im0, line_width=2, pil=True, example=str(names))
                if cfg.STRONGSORT.ECC:  # camera motion compensation
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    im_bb = len(det)
                    bb_all += im_bb

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort

                    t4 = time_sync()
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:

                        # Название файла для сохранения кадра: 
                        name = str(uuid.uuid4())
                        # im_save_path = 'RESULT/IMAGES/' + name + '.jpg'
                        
                        bboxes_per_row = []

                        # Итерир-ие по всем bbox кадра
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)): 
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            bbox_left = output[0] 
                            bbox_top = output[1]  
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            
                            name = str(uuid.uuid4())
                            im_save_path = os.path.join(images_fold, name + '.jpg')

                            # Отдельная проверка для столбов
                            # if (c == 4) & (conf < 0.8): 
                            #     continue 
                                
                            # # Для дорожной разметки: 
                            # if (c == 7) & (conf < 0.45):
                            #     continue

                            # top_pos, low_pos = output[3], output[1]

                            # k_top = 25
                            # k_low = 20

                            # # Итерирование по интервалам с коэфф-ами, определение 
                            # # наиблизжайшего уровня для верхней и нижней границ bbox'а: 
                            # for interv in coeffs.keys(): 
                            #     if (interv[0] <= top_pos <= interv[2]): 
                            #         if top_pos >= interv[1]:
                            #             k_top = coeffs.get(interv)[1]
                            #         else: k_top = coeffs.get(interv)[0]
                                
                            #     if (interv[0] <= low_pos <= interv[2]): 
                            #         if low_pos >= interv[1]:
                            #             k_low = coeffs.get(interv)[1]
                            #         else: k_low = coeffs.get(interv)[0]

                            # Инициализация объекта: 
                            bbox = {}
                            bbox['Object_id'] = f'{id}-{conf:.2f}' 
                            bbox['Lower_left_corner'] = [int(output[0]), int(output[1])]
                            bbox['Class_num'] = c
                            bbox['Bbox_width'] = int(bbox_w)
                            bbox['Bbox_height'] = int(bbox_h)

                            # Добавление bbox: 
                            bboxes_per_row.append(bbox)

                            bbox['Image_id'] = name 

                            # Масштабирование в реальный размер: 
                            label = translates[c]
                            # if c == 0: 
                            #     bbox['Length_mm'] = int((k_top+k_low) / 2 * (bbox_w**2 + bbox_h**2)**(1/2))
                            #     bbox['Width_mm'] = 0
                            # elif c in [1, 2, 3]: 
                            #     bbox['Length_mm'] = int(bbox_h * (k_top+k_low) / 2)
                            #     bbox['Width_mm'] = int(bbox_w * (k_top+k_low) / 2)

                            # if save_txt:
                            #     # to MOT format
                            #     # Write MOT compliant results to file
                            #     with open(txt_path + '.txt', 'a') as f:
                            #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                            #                                     bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                
                                if label in translates:
                                    bboxes = [torch.tensor(i) for i in bboxes]
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                    # im_save_path
                                    if save_crop:
                                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                        
                            # Сохранение кадров с детекцией: 
                            im = annotator.result() # frame 
                            cv2.imwrite(im_save_path, im)    
                        # 1 list is 1 row of inp txt: 
                        bboxes_list.append(bboxes_per_row)

                    # im = annotator.result() # frame 
                    # cv2.imwrite(im_save_path, im)   
                    t1, t2 = t3 - t2, t5 - t4
                    LOGGER.info(f'{s}Done. YOLO:({t1:.3f}s), StrongSORT:({t2:.3f}s)')
                    LOGGER.info('Объектов на кадре: {}; всего за видео: {}'.format(im_bb, bb_all))
                    # t1_all += t1
                    # t2_all += t2
                    # LOGGER.info(F'Среднее время YOLO: {t1_all / frame_idx}, StrongSORT: {t2_all / frame_idx}')

                else:
                    bboxes_list.append([])
                    strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_vid:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]
     

    if gps_txt_source:
        with open(gps_txt_source, 'r') as f: 
            txt_lines = f.readlines()

    # gps = 'RESULT/GPS_FILE.json'
    gps = f'RESULT/{vid_name}/' + f'{vid_name}.json'

    # new_js = str(gps) + '.json'
    with open(gps, 'w') as f: 
        list_to_json = []

        if gps_txt_source:
            # Запись bboxes в json с соотв. информацией: 
            step = int((fps/k) / inst_per_second) # Шаг нумерации
            count = 0

            for line in txt_lines: # итер. по bboxes (в виде строки) за кадр
                line = line.split()
                bbox_date, bbox_time = line[0], line[1]

                # Все bboxes за текушую строку входного ф.: 
                entry_bboxes = bboxes_list[count] if (0 < count < len(bboxes_list)) else None
                if entry_bboxes:
                    for bbox in entry_bboxes: 
                        bbox['Class_name'] = labls.get(bbox['Class_num']).encode('utf-8').decode('utf-8')

                        if len(line) == 5: 
                            bbox['GPS_coordinates'] = (line[-2], line[-1])
                            bbox['Date_of_detection'] = bbox_date
                            bbox['Time_of_detection'] = bbox_time 
                        else:
                            bbox['GPS_coordinates'] = 'incorrect_input_data'
                        list_to_json.append(bbox)
                count += step
        else: 
            for bboxes_row in bboxes_list:
                for n, bbox in enumerate(bboxes_row): 
                    bbox['Class_name'] = labls.get(bbox['Class_num']).encode('utf-8').decode('utf-8')
                    list_to_json.append(bbox)

        # prev_data_js.update(list_to_json)
        json.dump(list_to_json, f, indent=4, ensure_ascii=False)
    
    # Вычисления суммарных размерностей:

    abs_path = os.path.abspath(gps)
    data_pd = pd.read_json(abs_path, encoding="utf-8").copy()

    if len(data_pd) > 0: 

        data_pd['Id_main'] = data_pd['Object_id'].apply(lambda string: string.split('-')[0])
        new_vals = data_pd.groupby('Id_main').median()
        # new_vals['Coverage_area_m'] = new_vals['Length_mm'] * new_vals['Width_mm'] / (10**6) 

        # cols = ['Coverage_area_m', 'Width_mm', 'Length_mm']
        # new_vals[new_vals['Class_num'] != '1'][cols] = ''
        # new_vals[new_vals['Class_num'] != '2'][cols] = ''
        # new_vals[new_vals['Class_num'] != '3'][cols] = ''


        data_pd = data_pd.drop(
            columns=['Class_num']
            ).merge(new_vals, on='Id_main')
        
        # Информация о границах bbox'ов: 
        # data_pd.rename(columns={'Bbox_width_x': 'Bbox_width', 'Bbox_height_x': 'Bbox_height'}, inplace=True)

        # Исключ. повторений одного и того же объекта: 
        data_pd.drop_duplicates(subset='Id_main', inplace=True)

        # data_pd.drop(columns=['Id_main', 'Bbox_width_y', 'Bbox_height_y'], inplace=True)


        data_pd['Class_num'] = data_pd['Class_num'].astype(int)

        # Удаление лишних изображений в RESULT/IMAGES: 
        important_images = set(data_pd['Image_id'] + '.jpg')
        for file in os.listdir(images_fold): 
            if not file in important_images: 
                os.remove(os.path.join(images_fold, file))

        # # Площадь перекрытия для не-трещин: 
        # cols = ['Coverage_area_m', 'Width_mm', 'Length_mm']
        # data_pd[(data_pd['Class_name'] != 'large_pit') & ()][cols] = ''
        # data_pd[data_pd['Class_name'] != 'small_pit'][cols] = ''
        # data_pd[data_pd['Class_name'] != 'patch'][cols] = ''

        # data_pd.to_json('RESULT/GPS_FILE.JSON', orient='records', indent=4)

        # Общая длина трещин и общая площадь не-трещин: 
        # groups = {pit: data_pd[data_pd['Class_name'] == pit] for pit in data_pd['Class_name'].unique()}
        # if ('fissure' in groups.keys()): 
        #     val1 = groups.get('fissure')['Length_mm'].sum() / 1000 # общ. длина трещин
        # else: val1 = 0
        # val2_4 = {
        #     cl_name: groups.get(cl_name)['Coverage_area_m'].sum() 
        #     for cl_name in data_pd['Class_name'].unique() if cl_name in ('large_pit', 'small_pit', 'patch')
        #     }
        # result = {
        #     'Length_all (fissures), m': int(val1) if not None else 0,
        #     'Coverage_area_all (other), m^2': {
        #         'small_pits': val2_4.get('small_pit'), 
        #         'large_pits': val2_4.get('large_pit'),
        #         'patches': val2_4.get('patch') 
        #     } 
        # }

        # with open(f'RESULT/{vid_name}/INFO_ALL_{vid_name}.json', 'w') as f: 
        #     json.dump(result, f, indent=4)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'best_said.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    # argument for gps_txt_source, step_frame
    parser.add_argument('--gps-txt-source', type=str) 
    parser.add_argument('--step_frame', type=int) 
    parser.add_argument('--images_fold', type=str)
    parser.add_argument('--vid_name', type=str)


    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280, 1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image') 
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)