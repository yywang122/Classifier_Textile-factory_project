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
    
    for root, dirs, files in os.walk(input_folder):
        for subdir in dirs:
            input_subfolder = os.path.join(root, subdir)

            
            output_subfolder = os.path.join(output_folder, os.path.relpath(input_subfolder, input_folder))
            un_subfolder = os.path.join(unout_folder, os.path.relpath(input_subfolder, input_folder))

           
            os.makedirs(output_subfolder, exist_ok=True)
            os.makedirs(un_subfolder, exist_ok=True)

            image_files = [f for f in os.listdir(input_subfolder) if f.endswith('.jpg')]

            
            for image_file in image_files:
                try:
                    
                    roi_info = extract_roi_from_filename(image_file)
                    
                    

                    if roi_info:
                        input_image_path = os.path.join(input_subfolder, image_file)
                        image = cv2.imread(input_image_path)
                        min_row, max_row = roi_info
                        cropped_image = image[min_row:max_row, 0:image.shape[1], :]
                        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
                        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                        sobel_y = np.uint8(np.absolute(sobel_y))
                        _, binary_y = cv2.threshold(sobel_y, 30, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(binary_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        max_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(max_contour)
                
                        center_x = x + w // 2
                        crop_size = 380

                        crop_x = max(center_x - crop_size // 2, 0)
                        
                        if center_x>cropped_image.shape[1]- crop_size // 2:
                            
                            crop_x=crop_x-(center_x - (cropped_image.shape[1]- crop_size // 2))
                        
                        
                        cropped_image2 = cropped_image[0:max_row, crop_x:crop_x + crop_size, :]
                        
                        if cropped_image2.shape[0]==380 and cropped_image2.shape[1]==380:
                            print('yyyyyyy',cropped_image2.shape)
                            output_image_path = os.path.join(output_subfolder, image_file)

                           
                            cv2.imwrite(output_image_path, cropped_image2)
                        else:
                            print('image_file',image_file)
                            print('crop_x',crop_x)
                            print('center_x',center_x)
                
                            print('cropped_image',cropped_image.shape)
                            print('cropped_image2',cropped_image2.shape)
                            
                            un_image_path = os.path.join(un_subfolder, image_file)

                            cv2.imwrite(un_image_path, cropped_image2)

                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
                    continue

                    
input_parent_folder = '/home/cluster/data/fatory_training_data/factory_train_c'
output_parent_folder = '/home/cluster/data/fatory_training_data/factory_train_c_roi'
unout_folder = '/home/cluster/data/fatory_training_data/failed_roi'
crop_and_save_images(input_parent_folder, output_parent_folder,unout_folder)
