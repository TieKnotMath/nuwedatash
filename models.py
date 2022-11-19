import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, Rescaling, MaxPooling2D, BatchNormalization, LeakyReLU, Convolution2D, RandomFlip, RandomCrop, RandomTranslation, RandomRotation, RandomHeight, RandomZoom, RandomWidth, RandomContrast, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import pandas as pd
import tensorflow as tf
import json
import numpy as np



def resnet_50_custom():
    image_size=224
    metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
    resnet50_model = resnet50.ResNet50(weights="imagenet", include_top = False, input_shape=(image_size, image_size, 3))
    resnet50_model.trainable = True

    model = Sequential()
    model.add(Lambda(resnet50.preprocess_input, input_shape=(image_size, image_size, 3)))
    model.add(resnet50_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(3, activation='softmax'))

    return model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.999), metrics=["accuracy", metric])



def load_data():
    TRAINDIR = './trainv3'
    VALDIR = './testv2'
    TESTDIR = './testv2'
    train_dataset = image_dataset_from_directory(
        TRAINDIR, 
        image_size = (224, 224),
        batch_size = 32, 
        label_mode = 'categorical')
    val_dataset = image_dataset_from_directory(
        VALDIR, 
        image_size = (224, 224),
        batch_size = 32, 
        label_mode = 'categorical')
    test_dataset = image_dataset_from_directory(
        TESTDIR, 
        image_size = (224, 224),
        batch_size = 32, 
        label_mode = 'categorical')
    return train_dataset, val_dataset, test_dataset


def predict():
    df = pd.read_csv('test.csv')
    metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
    model = tf.keras.models.load_model('Resnet50_stack_f182.h5')
    lista = []
    for i in df.example_path:
        img = image.load_img(i, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        preds = model.predict(x)[0]
        lista.append(list(preds).index(max(list(preds))))
    df['prediction'] = lista
    df['name'] = df.example_path.apply(lambda x: x.split('/')[-1].split('.')[0]) 
    res = {}
    res["target"] = {}
    for idx, item in df.iterrows():
        res["target"][str(item.loc['name'])] = item.loc['prediction']
    with open("predictions.json", "w") as outfile:
        json.dump(res, outfile, indent=4)

    return res

            