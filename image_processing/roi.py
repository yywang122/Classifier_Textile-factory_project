import cv2
import os
import re
from shutil import copytree
import matplotlib.pyplot as plt
def extract_roi_from_filename(filename):
    pattern = re.compile(r'(south|north)(\d+)_(\d+)_(\d+)_CAM(\d+)_NG\((\d+\.\d+).*\)\.jpg')
    match = pattern.match(filename)

    if match:
        direction = match.group(1)

        if "CAM2" in filename:
            if "south" in direction:
                return (550, 930)
            elif "north" in direction:
                return (500, 880)
        elif "CAM1" in filename:
            if "south" in direction:
                return (300, 680)
            elif "north" in direction:
                return (435, 815)

    return None


def crop_and_save_images(input_folder, output_folder,unout_folder):
    # 遍历输入文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(input_folder):
        for subdir in dirs:
            input_subfolder = os.path.join(root, subdir)

            # 创建输出子文件夹的完整路径
            output_subfolder = os.path.join(output_folder, os.path.relpath(input_subfolder, input_folder))
            un_subfolder = os.path.join(unout_folder, os.path.relpath(input_subfolder, input_folder))

            # 创建输出子文件夹
            os.makedirs(output_subfolder, exist_ok=True)
            # 创建输出子文件夹
            os.makedirs(un_subfolder, exist_ok=True)

            # 遍历子文件夹中的所有图像文件
            image_files = [f for f in os.listdir(input_subfolder) if f.endswith('.jpg')]

            # 遍历子文件夹中的每个图像文件
            for image_file in image_files:
                try:
                    # 提取ROI信息
                    #print("########################")
                    #image = cv2.imread(image_file)
                    roi_info = extract_roi_from_filename(image_file)
                    
                    

                    if roi_info:# 读取图像
                        # 构建图像文件的完整输入路径
                        input_image_path = os.path.join(input_subfolder, image_file)

                        # 读取图像
                        image = cv2.imread(input_image_path)
                        #print(image.shape)

                        # 根据提取的ROI信息裁剪图像的区域
                        min_row, max_row = roi_info
                        #print('~~~~~~~~~~~',(min_row,max_row))
                        cropped_image = image[min_row:max_row, 0:image.shape[1], :]
                        #print(cropped_image.size)
                        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 

                        # 使用 Sobel 運算子计算垂直方向的梯度
                        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                        # 将梯度值的绝对值转换为 8 位无符号整数
                        sobel_y = np.uint8(np.absolute(sobel_y))

                        # 二值化处理，提取垂直方向的边缘
                        _, binary_y = cv2.threshold(sobel_y, 30, 255, cv2.THRESH_BINARY)

                        # 找到轮廓
                        contours, _ = cv2.findContours(binary_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                        # 找到最大轮廓
                        max_contour = max(contours, key=cv2.contourArea)

                        # 获取最大轮廓的外接矩形
                        x, y, w, h = cv2.boundingRect(max_contour)
                        
                        # 获取外接矩形的中心点
                        center_x = x + w // 2
                        #center_y = y + h // 2

                        # 设置裁剪区域大小
                        crop_size = 380

                        # 计算裁剪区域的位置
                        crop_x = max(center_x - crop_size // 2, 0)
                        
                        if center_x>cropped_image.shape[1]- crop_size // 2:
                            
                            crop_x=crop_x-(center_x - (cropped_image.shape[1]- crop_size // 2))
                        
                        #crop_y = max(center_y - crop_size // 2, 0)

                        # 裁剪图像
                        cropped_image2 = cropped_image[0:max_row, crop_x:crop_x + crop_size, :]
                        #plt.imshow(cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2RGB))
                        #plt.title('Contours Detected')
                        #plt.axis('off')
                        #plt.show()
                        
                        if cropped_image2.shape[0]==380 and cropped_image2.shape[1]==380:
                            print('yyyyyyy',cropped_image2.shape)
                            output_image_path = os.path.join(output_subfolder, image_file)

                            # 保存裁剪后的图像到输出子文件夹
                            cv2.imwrite(output_image_path, cropped_image2)
                        else:
                            print('image_file',image_file)
                            print('crop_x',crop_x)
                            print('center_x',center_x)
                
                            print('cropped_image',cropped_image.shape)
                            print('nnnnnnnnnn',cropped_image2.shape)
                            
                            un_image_path = os.path.join(un_subfolder, image_file)

                            # 保存裁剪后的图像到输出子文件夹
                            cv2.imwrite(un_image_path, cropped_image2)

                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
                    continue
                #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    
input_parent_folder = '/home/cluster/data/fatory_training_data/factory_train_c'
output_parent_folder = '/home/cluster/data/fatory_training_data/factory_train_c_roi'
unout_folder = '/home/cluster/data/fatory_training_data/failed_roi'
# 执行裁剪和保存操作
crop_and_save_images(input_parent_folder, output_parent_folder,unout_folder)
