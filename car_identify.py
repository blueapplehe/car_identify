#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,shutil,sys
sys.path.append('/data/py/lib/') 
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.applications import xception
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
import time
original_dataset_dir='/data/keras/download/qiche/all'
base_dir='/data/keras/download/qiche'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')
mod_names=["速腾","朗逸","捷达","迈腾","高尔夫","桑塔纳","帕萨特",
          "思域","本田CR-V","雅阁",
          "雷凌","威驰","凯美瑞","卡罗拉",
          "轩逸","天籁",
          "福克斯","福睿斯","蒙迪欧"]
mod_names=["速腾","迈腾","雷凌","卡罗拉","凯美瑞","天籁","雅阁","朗逸",
          "威驰","福克斯","福睿斯","蒙迪欧","轩逸","帕萨特","途观","飞度","锋范"]
epochs=10
nnn=len(mod_names)
img_height=384
img_width=512
img_height=299
img_width=299


# In[2]:


from keras.preprocessing.image import ImageDataGenerator
base_dir='/data/keras/download/qiche'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
     zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
#train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height,img_width),
    batch_size=10,
    class_mode='categorical'   
)

validation_generator=test_datagen.flow_from_directory(   
    validation_dir,
    target_size=(img_height,img_width),
    batch_size=5,
    class_mode='categorical'
)


# In[3]:





#contv_base=VGG16(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
# contv_base.summary()
# contv_base.trainable=True
# set_trainable=False
# for layer in contv_base.layers:
#     if layer.name=='block1_conv5':
#         set_trainable=True
#     if set_trainable:
#         layer.trainable=True
#     else:
#         layer.trainable=False


#contv_base=keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(img_height,img_width,3))
       

# contv_base=xception.Xception(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))
# contv_base.summary()
# for layer in contv_base.layers:
#      layer.trainable = False
# for i, layer in enumerate(contv_base.layers):
#     print(i, layer.name)


# model=models.Sequential()
# model.add(contv_base)
# model.add(layers.Flatten())
# #model.add(layers.Dropout(0.1))
# model.add(layers.Dense(256,activation='relu'))
# model.add(layers.Dense(nnn,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])


# In[4]:


# history=model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.n/train_generator.batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.n/validation_generator.batch_size
# )


# In[5]:


# for layer in model.layers[:1]:
#    layer.trainable = False
# for layer in model.layers[1:]:
#    layer.trainable = True
# model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
# history=model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.n/train_generator.batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.n/validation_generator.batch_size
# )


# In[ ]:





# In[6]:



# 构建不带分类器的预训练模型
base_model = xception.Xception(weights="imagenet",include_top=False,input_shape=(img_height,img_width,3))

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(256, activation='relu')(x)

# 添加一个分类器，假设我们有200个类
predictions = Dense(nnn, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer=optimizers.RMSprop(lr=1e-3), loss='categorical_crossentropy',metrics=['acc'])

# 在新的数据集上训练几代
# history=model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.n/train_generator.batch_size,
#     epochs=5,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.n/validation_generator.batch_size
# )

# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
# for layer in model.layers[:100]:
#    layer.trainable = False
for layer in model.layers[0:]:
   layer.trainable = True

# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调
from keras.optimizers import SGD
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',metrics=['acc'])

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
history=model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n/train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.n/validation_generator.batch_size
)
time_t=time.strftime("%m%d%H%M", time.localtime()) 
model.save('/data/keras/models/%s.h'%time_t)


# In[10]:


history=model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n/train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.n/validation_generator.batch_size
)


# In[11]:


import matplotlib.pyplot as plt
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


# In[12]:


from keras.preprocessing import image
import numpy as np
import cv2
from mytool import MyTool
test_imgs=['/data/keras/download/qiche/timg2.jpg',
          '/data/keras/download/qiche/timg.jpg',
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




