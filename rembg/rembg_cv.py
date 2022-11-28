from rembg import remove
import cv2

input_path = '/home/vk/Desktop/CV_BackgroundRemoval/rembg/6.jpg'
output_path = '/home/vk/Desktop/CV_BackgroundRemoval/rembg/rembg_output2.png'

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)