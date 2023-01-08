import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


SAVE_IMAGES = 'save_images/'
DETECT_IMAGES = 'detection_images/'
# замена int классов на значения
dict_repl_classes = {
    0: 'low_curb', 
    1: 'faulty_drainage', 
    2: 'roadside_problems', 
    3: 'rutting', # колейность
    4: 'sweating_bitumen', # выпотевающий битум 
    5: 'trash', 
    6: 'roadside_problems', 
    7: 'rutting', # колейность
    8: 'sweating_bitumen', # выпотевающий битум 
    9: 'trash', 
}

# English version vs Russian version
change_value_labels = {
    'low_curb': 'Занижение обочины', 
    'faulty_drainage': 'Неисправный дренаж', 
    'roadside_problems': 'Просадки на обочине', 
    'rutting': 'Гребенка', 
    'sweating_bitumen': 'Выпотевающий битум', # выпотевающий битум 
    'trash': 'Посторонний предмет'
} 
# change_value_labels = {
#     'fissure': 'Трещина', 
#     'large_pit': 'Большая яма', 
#     'small_pit': 'Маленьькая яма', 
#     'patch': 'Заплатка'
# }

# словарь с вероятностями
dict_proba_classes = {
    0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5,
    4: 0.3, 5: 0.5, 6: 0.5, 7: 0.4,
    8: 0.5, 9: 0.5, 10: 0.9,
    11: 0.5, 12: 0.5, 13: 0.5, 14: 0.3
}

@torch.no_grad()
def detect(
    source,
    model,
    imgsz=(720, 720),  # inference size (height, width)
    conf_thres=0.3,  # confidence threshold
    iou_thres=0.4,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=True,  # visualize features
    update=False,  # update all models
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    """
    Функция для представления результата.
    """
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Directories
    save_dir = increment_path(DETECT_IMAGES, True)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    #device = select_device(device)
    #model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half) # data=data,
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(SAVE_IMAGES + source, img_size=imgsz, stride=stride)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    all_labels = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(DETECT_IMAGES, True) 
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=True, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    print(f"ВЕРОЯТНОСТИ, {conf}, {cls}")
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        predict_val = dict_proba_classes.get(c)
                        if conf.item() > predict_val:
                            #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label = change_value_labels[names[c].replace('dustbin', 'rubbish').replace('graffiti', 'advertising')]
                            all_labels.append(label)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
    return all_labels