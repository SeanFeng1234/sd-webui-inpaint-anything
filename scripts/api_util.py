import numpy as np
import cv2
import os

def crop_face(input_folder_path, output_folder_path):
    # 加载面部识别模型
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # image =input_folder_path

    image_path = input_folder_path
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测面部
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # 无法识别面部的图片
    if len(faces) == 0:
        print(f"No face found in {image_path}")
        return

    if len(faces) > 0:
        # 取第一个脸部位置，这里假设一张图片只有一个脸部特征
        # x、y 为人脸的像素位置，w、h 为人脸的宽度和高度。
        x, y, w, h = faces[0]
        # 确定最大正方形的位置
        # 原图片竖方向长，截取正方形长度为原图横方向长，square_size截取正方形的长度
        if height > width:
            square_size = width
            x1 = 0
            x2 = square_size
            # 原图面部靠上
            if y < square_size / 2:
                y1 = 0
                y2 = square_size
            # 原图面部靠下
            else:
                y1 = int(square_size / 2)
                y2 = height
        # 原图片是横方向长，截取正方形长度为原图竖方向长度
        else:
            square_size = height
            y1 = 0
            y2 = square_size
            # 原图面部靠右
            if x < square_size / 2:
                x1 = 0
                x2 = square_size
            # 原图面部靠左
            else:
                x1 = int(square_size / 2)
                x2 = square_size

        # 根据最大正方形位置裁剪图片并保存
        cropped_img = img[y1:y2, x1:x2]
        # 调整图像大小为512x512
        resized = cv2.resize(cropped_img, (square_size, square_size), interpolation=cv2.INTER_AREA)
        output_path = os.path.join(output_folder_path, "11.jpg")
        cv2.imwrite(output_path, resized)

        return output_path