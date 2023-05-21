from Models import SSDLikeModel
from Utils import load_input_image, draw_predictions

import matplotlib.pyplot as plt
import numpy as np
import os

all_images_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\JPEGImages'
model_path = r"D:\E\freelancer projects\Advance-Computer-Vision\Pascal-Voc-Object-Detection\Models\ssd_model.h5"
pre_parsed_csv_path = r'./Utils/data_details.csv'
img_size = (300, 300)

images_list = [i for i in map(lambda x: os.path.join(all_images_path, x), os.listdir(all_images_path))]
images_mini = images_list[900:940]

images = []
for i in images_mini:
    images.append(load_input_image(i, img_size=img_size))
images = np.array(images)

o, data_details = SSDLikeModel.predict_from_load_model(images, model_path, pre_parsed_csv_path=pre_parsed_csv_path)

for x in range(0, 40, 4):
    plt.figure()
    for it, i in enumerate(range(0+x, 4+x)):
        img = images[i]

        plt.subplot(2, 2, it+1)

        draw_predictions(o[i], img, data_details, img_size, conf_threshold=0.97)
    plt.tight_layout()

plt.show()
