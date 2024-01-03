import cv2
import os
#Sticky Hairiness Silk_scraps

# 輸入資料夾和輸出資料夾
input_folder = '/home/cluster/data/fatory_training_data/factory_data_b_super/val/Sticky'
output_folder = '/home/cluster/data/fatory_training_data/factory_data_b_super_c/val/Sticky'
# 設定新的裁剪大小
crop_width = 380
crop_height = 380


# 確保輸出資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 列出輸入資料夾中的所有圖片
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

current_image_index = 0

while current_image_index < len(image_files):
    image_file = image_files[current_image_index]

    # 讀取圖片
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)

    # 建立視窗，顯示圖片
    cv2.imshow("Select Center", img)
    key = cv2.waitKey(0)
    if key == ord('a'):
        # 上一張圖片
        current_image_index = max(0, current_image_index - 1)

    elif key == ord('d'):
        # 下一張圖片
        current_image_index = min(len(image_files) - 1, current_image_index + 1)
    # 要求使用者選擇中心點
    else:
        center_point = cv2.selectROI("Select Center", img, fromCenter=True, showCrosshair=True)

    # 取得中心點座標
        center_x, center_y, _, _ = center_point

    # 裁剪區域的左上角座標
        x = int(center_x - crop_width / 2)
        y = int(center_y - crop_height / 2)

    # 裁剪區域的右下角座標
        x_end = x + crop_width
        y_end = y + crop_height

    # 裁剪圖片
        cropped_img = img[y:y_end, x:x_end]
    # 顯示裁剪後的圖片
        cv2.destroyAllWindows()
        cv2.imshow("Cropped Image", cropped_img)
        key = cv2.waitKey(0)

        if key == ord('s'):
        # 決定保存的檔案名稱，例如 "cropped_image_1.jpg"
            output_name = f"cropped_{image_file.split('.jpg')[0]}_{current_image_index + 1}.jpg"
        
        # 避免檔案名稱重複，若已存在則逐一增加索引
            count = 1
            while os.path.exists(os.path.join(output_folder, output_name)):
                output_name = f"cropped_{image_file.split('.jpg')[0]}_{current_image_index + 1}_{count}.jpg"
                count += 1

        # 另存新的圖片到輸出資料夾
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved {output_path}")
    

    # 關閉裁剪後的圖片視窗
        cv2.destroyWindow("Cropped Image")
	
print("圖片處理完成。")
cv2.destroyAllWindows()




