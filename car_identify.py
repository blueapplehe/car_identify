#!/usr/bin/env python
# coding: utf-8
import os,shutil,sys
sys.path.append('/data/py/lib/') 
import keras
import time
from keras import models
from keras import layers
from keras import optimizers
#from keras.applications import VGG16
from keras.applications import xception
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
####定义一些常用的训练调整参数###########
epochs=5            #定义训练轮数
batch_size=20       #每批数量
lock_layer_num=0;   #锁住的层数
lr=1e-4             #学习率
dense_num=256       #连接层数量
pre_train_epochs=1  #预训练轮数,0表示不进行预训练
img_height=299      #训练图片高度
img_width=299       #训练图片宽度
is_load_model=False #是否加载自己训练的历史模型
##########################

base_dir='/data/keras/download/qiche'#汽车图片根目录
train_dir=os.path.join(base_dir,'train')#汽车图片训练目录
validation_dir=os.path.join(base_dir,'validation')#汽车图片验证目录
test_dir=os.path.join(base_dir,'test')#汽车图片测试目录
#精选一些品牌的汽车种类，引入更多的品牌的种类，不会大影响识别准确率，放心推广到更多的品牌和车型，
#这里不演示更多的品牌，是因为我的显卡太烂了，图片太多，训练速度有点满
mod_names=["速腾","迈腾","雷凌","卡罗拉","凯美瑞",
           "天籁","雅阁","朗逸","威驰","福克斯",
           "福睿斯","蒙迪欧","轩逸","帕萨特","途观",
           "飞度","锋范"]

mod_num=len(mod_names)#汽车车型总数


#使用图片数据增强，降低拟合的有效手段
train_datagen=ImageDataGenerator(  
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#验证，测试数据不能进行数据增强
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical'   
)

validation_generator=test_datagen.flow_from_directory(   
    validation_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


if is_load_model is False:
    # 构建不带分类器的预训练模型
    base_model = xception.Xception(weights="imagenet",include_top=False,input_shape=(img_height,img_width,3))

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(dense_num, activation='relu')(x)

    # 添加一个分类器
    predictions = Dense(mod_num, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 锁住所有 Xception 的卷积层
    for layer in base_model.layers:
        layer.trainable = False

    #预训练
    if pre_train_epochs>0:
        model.compile(optimizer=optimizers.rmsprop_v2.RMSProp(learning_rate=1e-3), loss='categorical_crossentropy',metrics=['acc'])
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n/train_generator.batch_size,
            epochs=pre_train_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.n/validation_generator.batch_size
        )

    # 现在顶层应该训练好了，开始微调 Xception的卷积层。
    # 锁住底下的几层，然后训练其余的顶层。
    # 看看每一层的名字和层号，看看我们应该锁多少层呢：
    # for i, layer in enumerate(base_model.layers):
    #    print(i, layer.name)

    # 锁住的层数
    for layer in model.layers[:lock_layer_num]:
       layer.trainable = False
    for layer in model.layers[lock_layer_num:]:
       layer.trainable = True

    # 设置一个很低的学习率，使用 SGD 来微调
    model.compile(optimizer=optimizers.rmsprop_v2.RMSprop(lr=lr), loss='categorical_crossentropy',metrics=['acc'])

    # 继续训练模型
    history=model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n/train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n/validation_generator.batch_size
    )
    #保存训练好的模型
    time_t=time.strftime("%m%d%H%M", time.localtime()) 
    model.save('/data/keras/models/%s.h'%time_t)


#显示训练过程中精度变化
if is_load_model is False:
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.legend()
    plt.figure()
    plt.show()

#显示测试结果
from keras.preprocessing import image
import numpy as np
import cv2
from tools.mytool import MyTool
test_imgs=['/data/keras/download/qiche/timg2.jpg',
          '/data/keras/download/qiche/su1.jpg',
          '/data/keras/download/qiche/su2.jpg',
          '/data/keras/download/qiche/su3.jpg',
          '/data/test/su21.jpg',
          '/data/test/su22.jpg',
          '/data/test/su23.jpg',
          '/data/test/su24.jpeg',
          '/data/test/su25.jpeg',
          '/data/test/su26.jpeg',
           '/data/test/mt20.jpg',
           '/data/test/mt21.jpg',
           '/data/test/mt22.jpg',
           '/data/test/mt23.jpg',
           '/data/test/kll20.jpg',
            '/data/test/kll21.jpg',
            '/data/test/kll22.jpg',
            '/data/test/kll23.jpg',
            '/data/test/kll24.jpg',
          ]

for img_path in test_imgs:
    #img = image.load_img(img_path, target_size=(img_height, img_width))
    img =cv2.imread(img_path)
#     plt.imshow(img)
#     plt.show()
    img=MyTool.cro_img(img,img_height,img_width)
    plt.imshow(img)
    plt.show()
    
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    paixu=dict(zip(train_generator.class_indices,preds[0]))
    paixu= sorted(paixu.items(), key=lambda x: x[1], reverse=True)
    print(paixu)