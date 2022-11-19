import pandas as pd
from PIL import Image
import numpy as np
import os
import shutil
import cv2

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf



def data_augmentation(path_train_csv):
        np.random.seed(1984)
        os.mkdir('testv2')
        os.mkdir('testv2/0')
        os.mkdir('testv2/1')
        os.mkdir('testv2/2')
        os.mkdir('trainv3')
        os.mkdir('trainv3/0')
        os.mkdir('trainv3/1')
        os.mkdir('trainv3/2')

        df_train = pd.read_csv(path_train_csv)

        for idx, item in df_train.iterrows():
            label = str(item.loc['label'])
            name = item.loc["example_path"].split('/')[-1]
            shutil.copy(item.loc["example_path"], 'trainv3/'+label+'/'+name)

        
        for label in os.listdir('trainv3'):
            for image in os.listdir('trainv3/'+str(label)):
                root = 'trainv3/'+str(label)+'/'+image
                if np.random.rand()< 0.1:
                    shutil.move(root, 'testv2/'+str(label)+'/'+image)
        lista_images = []
        for label in os.listdir('trainv3'):
            for image in os.listdir('trainv3/'+label):
                lista_images.append('trainv3/'+label+'/'+image)
        
        for i in lista_images:
            base = i.replace('trainv3', 'trainv3')
            last = base.split('/')[-1].split('.')[0]
            image = Image.open(i)
            image.rotate(90).save("/".join(base.split('/')[:-1])+'/'+last+'_r1.png')
            image.rotate(180).save("/".join(base.split('/')[:-1])+'/'+last+'_r2.png')
            image.rotate(270).save("/".join(base.split('/')[:-1])+'/'+last+'_r3.png')
            Image.fromarray(apply_brightness_contrast(np.array(image.rotate(90)), brightness = 13, contrast = 0)).save("/".join(base.split('/')[:-1])+'/'+last+'_b1.png')
            Image.fromarray(apply_brightness_contrast(np.array(image.rotate(180)), brightness = -55, contrast = 0)).save("/".join(base.split('/')[:-1])+'/'+last+'_b2.png')
            Image.fromarray(apply_brightness_contrast(np.array(image.rotate(270)), brightness = 0, contrast = 5)).save("/".join(base.split('/')[:-1])+'/'+last+'_c2.png')