import glob as gl
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# image_count = len(list(gl.glob('assets/**/*.[jJ][pP]*[gG]')))
# print(f'{image_count} examples of numbers')

# categories = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

# for category in categories:
#     count = len(list(gl.glob(f'assets/{category}/*.[jJ][pP]*[gG]')))
#     print(f"{category} count = {count}")

batch_size = 32
class_count = 10

img_height = 28
img_width = 28

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'assets',
    subset = 'training',
    validation_split = 0.2,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'assets',
    subset = 'validation',
    validation_split = 0.2,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
# print(f'class names: {class_names}')

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(class_count)
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.summary()
epochs=20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from sklearn.metrics import classification_report

def evaluate_model(val_ds,model):
    y_pred=[]
    y_true=[]

    for batch_images,batch_labels in val_ds:
        predictions=model.predict(batch_images,verbose=0)
        y_pred=y_pred+np.argmax(tf.nn.softmax(predictions),axis=1).tolist()
        y_true=y_true+batch_labels.numpy().tolist()
    print(classification_report(y_true,y_pred))


evaluate_model(val_ds,model)

model.save('model.keras')