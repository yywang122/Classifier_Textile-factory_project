

import os
import shutil
import random

def split_data(input_folder, output_folder, train_ratio, val_ratio):
    
    categories = os.listdir(input_folder)

    for category in categories:
        category_path = os.path.join(input_folder, category)
        
        
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        
        random.shuffle(images)
        
       
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        
        
        train_images = images[:num_train]
        val_images = images[num_train:num_train+num_val]
        
        train_folder = os.path.join(output_folder, 'train', category)
        val_folder = os.path.join(output_folder, 'val', category)
        
        
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        for image in train_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(train_folder, image)
            shutil.copy(src, dst)

        for image in val_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(val_folder, image)
            shutil.copy(src, dst)

input_folder = '/home/cluster/data/factory_data_c_more'
output_folder = '/home/cluster/data/factory_train_c_more'
split_data(input_folder, output_folder, train_ratio=0.7, val_ratio=0.3)


