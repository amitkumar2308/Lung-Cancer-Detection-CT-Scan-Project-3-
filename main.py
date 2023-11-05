import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

inception_v3.summary()

for layer in inception_v3.layers[: -15]:
    layer.trainable = False
inception_v3.summary()

x = inception_v3.output
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(units=4, activation='softmax')(x)
model = Model(inception_v3.input, output)

model.summary()

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('C:\\Users\\amitk\\Desktop\\Cancer Detection\\Data\\train',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = train_datagen.flow_from_directory('C:\\Users\\amitk\\Desktop\\Cancer Detection\\Data\\test',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')

histry = model.fit_generator(training_set,
                            validation_data=test_set,
                            epochs=10,
                            steps_per_epoch=len(training_set),
                            validation_steps=len(test_set)
                            )

# plot the loss
plt.plot(histry.history['loss'], label='train loss')
plt.plot(histry.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(histry.history['accuracy'], label='train acc')
plt.plot(histry.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('inception_chest.h5')
y_pred=model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
y_pred

classes = ["AdenocarcinomaChest Lung Cancer ","Large cell carcinoma Lung Cancer" , "NO Lung Cancer/ NORMAL" , "Squamous cell carcinoma Lung Cancer"]
def predict_image(img):
    plt.figure(figsize=(40,8))
    print()
    print('-----------------------------------Chest Cancer Type Detection---------------------------------------------------')
    print()
    print('----------------------------------------------RESULT-------------------------------------------------------------')
    print()
    x=image.img_to_array(img)
    x=x/255
    plt.imshow(img)
    x=np.expand_dims(x,axis=0)
    #img_data=preprocess_input(x)

    print(classes[np.argmax(model.predict(x))])
im=image.load_img('/kaggle/input/chest-ctscan-images/Data/test/adenocarcinoma/000113 (7).png',target_size=(224,224))
predict_image(im)

