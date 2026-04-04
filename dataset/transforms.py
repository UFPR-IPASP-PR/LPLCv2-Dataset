import albumentations as A
import cv2
import json

def train_transforms(imgsz, config=None):
    if config is None:
        trs = A.Compose([
            A.PadIfNeeded(
                min_height=imgsz,
                min_width=imgsz,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                fill=127
            ),
            A.LongestMaxSize(
                max_size=imgsz,
                interpolation=cv2.INTER_CUBIC
            ),
            A.PadIfNeeded(
                min_height=imgsz,
                min_width=imgsz,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                fill=127
            ),
            A.ToFloat(),
            A.ToTensorV2()
        ])
    else:
        with open(config, "r") as fd:
            tr_dict = json.load(fd)
        trs = A.from_dict(tr_dict)
        
    return trs

def val_transforms(imgsz):
    trs = A.Compose([
        A.PadIfNeeded(
            min_height=imgsz,
            min_width=imgsz,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=127
        ),
        A.LongestMaxSize(
            max_size=imgsz,
            interpolation=cv2.INTER_CUBIC
        ),
        A.PadIfNeeded(
            min_height=imgsz,
            min_width=imgsz,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=127
        ),
        A.ToFloat(),
        A.ToTensorV2()
    ])
    return trs