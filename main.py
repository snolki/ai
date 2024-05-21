import tensorflow as tf
import cv2 
import os

train_dir = '/home/user3/Downloads/archive/leapgestrecog/leapGestRecog/'
validation_dir = '/home/user3/Downloads/archive/leapGestRecog/00/'

IMG_SIZE = (640, 240)
BATCH_SIZE = 1

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(640, 240, 3)),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(10, activation = 'softmax')
	
])

model.compile(
	optimizer = 'adam',
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics=['accuracy']
    )

model.fit(
    normalized_train_dataset,
    validation_data=normalized_validation_dataset,
    epochs=10
)

while(True): 
	
	ret, frame = vid.read() 

	cv2.imshow('frame', frame) 

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 

cv2.destroyAllWindows() 
