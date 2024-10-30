#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:07:03 2023

@author: cluster
"""

import os
import shutil

def collect_images_from_subdirectories(root_folder, target_folder, keyword):
    # 在目標資料夾中創建一個子目錄，用於存放匯集的影像
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遞迴搜尋資料夾
    for root, dirs, files in os.walk(root_folder):
        # 檢查資料夾名稱是否包含關鍵字
        if keyword in os.path.basename(root):
            # 遍歷資料夾，將影像複製到目標子目錄中
            for filename in files:
                if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    source_file = os.path.join(root, filename)
                    target_file = os.path.join(target_folder, filename)
                    shutil.copy2(source_file, target_file)




# Sticky 貼絲
# Silk_scraps 絲屑
# Hairiness 毛羽


if __name__ == "__main__":
    root_folder = "/home/cluster/Downloads/factory_0801"  # 替換為您的根目錄路徑
    target_folder = "/home/cluster/Downloads/factory_data/Silk_scraps"  # 替換為您的目標目錄路徑
    keyword = "絲屑"  # 替換為您要尋找的關鍵字

    collect_images_from_subdirectories(root_folder, target_folder, keyword)



