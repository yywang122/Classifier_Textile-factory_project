import os
import cv2
import numpy as np
import supervision as sv

import torch
import torch.nn as nn
import torchvision
from Tool_data.tool_file import *
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor,build_sam
#from segment_anything import *
import time
from tqdm import tqdm, trange

CUDA_VISIBLE_DEVICES=0,1,2,3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device)
sam_predictor = SamPredictor(sam)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

#CLASSES = ["dog","bird","wheeled vehicle","reptile","carnivore","insect","musical instrument","primate","fish"]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
NMS_THRESHOLD = 0.80

'''
dire1 ='/home/D000017983/mydata/build_data/data/IM9_original/val/08_fish'
outdir='/home/D000017983/mydata/build_data/data/IM9_original_val_mask_08'
outdir1='/home/D000017983/mydata/build_data/data/IM9_original_val_08'
'''
dire1 ='/home/cluster/data/IM9_original'
outdir='/home/cluster/data/IM9_original_sam4'
outdir1='/home/cluster/data/IM9_original_mask4'

file_list1 = get_file_list(dire1)
build_list = build_file(outdir,dire1)
build_list2 = build_file(outdir1,dire1)
for file1 in tqdm(file_list1):
    
    image=cv2.imread(file1)
    # detect objects
    if file1.split('/')[-2][3:]=='dog':
        CLASSES = ["dog"]
    elif file1.split('/')[-2][3:]=='bird':
        CLASSES = ["bird"]
    elif file1.split('/')[-2][3:]=='wheeled vehicle':
        CLASSES = ["wheeled vehicle"]
       
    elif file1.split('/')[-2][3:]=='reptile':
        CLASSES = ["reptile"]
        
    elif file1.split('/')[-2][3:]=='carnivore':
        CLASSES = ["carnivore"]
        
    elif file1.split('/')[-2][3:]=='insect':
        CLASSES = ["insect"]
        
    elif file1.split('/')[-2][3:]=='musical instrument':
        CLASSES = ["musical instrument"]
        
    elif file1.split('/')[-2][3:]=='primate':
        CLASSES = ["monkey"]
        
    elif file1.split('/')[-2][3:]=='fish':
        CLASSES=["fish"]
        
    detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD)
    
    
    detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy)
    
    binary_masks=[mask.astype(np.uint8)*255 for mask in detections.mask]
    
    for i,mask in enumerate(binary_masks):
        relative_path=os.path.relpath(file1,dire1)
        filename =os.path.splitext(relative_path)[0]+'.png'
        outpath=os.path.join(outdir1,filename)
        mask=cv2.resize(mask,(image.shape[:-1][1],image.shape[:-1][0]))
        cv2.imwrite(outpath, mask)
        
    
    mask_rgb=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    print('image_shape :',image.shape)
    
    
    #mask=cv2.resize(mask,image.shape[:-1])
    print('mask_shape:',mask.shape)
    
    
    print('mask_rgb_shape :',mask_rgb.shape)
    masked_image=cv2.bitwise_and(image, mask_rgb) 
    
    relative_path=os.path.relpath(file1,dire1)
    filename =os.path.splitext(relative_path)[0]+'.png'
    outpath=os.path.join(outdir,filename)
    cv2.imwrite(outpath, masked_image)
    print(file1)
    
    
    
















# convert detections to masks


#masks


'''
mask_rgb=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#foreground
masked_image=cv2.bitwise_and(image, mask_rgb)


# save the annotated grounded-sam image
cv2.imwrite("grounded_sam_n01664065_17447.png", masked_image)
'''
