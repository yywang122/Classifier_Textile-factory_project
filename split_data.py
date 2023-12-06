

import os
import shutil
import random
#def split_data(input_folder, output_folder, train_ratio, val_ratio, test_ratio):
def split_data(input_folder, output_folder, train_ratio, val_ratio):
    # 获取所有类别文件夹
    categories = os.listdir(input_folder)

    for category in categories:
        category_path = os.path.join(input_folder, category)
        
        # 获取该类别下的所有图像文件
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 打乱图像顺序
        random.shuffle(images)
        
        # 计算划分数量
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        
        # 划分图像
        train_images = images[:num_train]
        val_images = images[num_train:num_train+num_val]
        #test_images = images[num_train+num_val:]
        
        # 创建输出文件夹
        train_folder = os.path.join(output_folder, 'train', category)
        val_folder = os.path.join(output_folder, 'val', category)
        #test_folder = os.path.join(output_folder, 'test', category)
        
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        #os.makedirs(test_folder, exist_ok=True)
        
        # 移动图像到相应文件夹
        for image in train_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(train_folder, image)
            shutil.copy(src, dst)

        for image in val_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(val_folder, image)
            shutil.copy(src, dst)
'''
        for image in test_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(test_folder, image)
            shutil.copy(src, dst)
'''
# 输入和输出文件夹路径



input_folder = '/home/cluster/data/factory_data_c_more'
output_folder = '/home/cluster/data/factory_train_c_more'




# 调用函数进行划分
#split_data(input_folder, output_folder, train_ratio=0.6, val_ratio=0.4, test_ratio=0.2)
split_data(input_folder, output_folder, train_ratio=0.7, val_ratio=0.3)


