import tensorflow as tf
import cv2 

model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(128,128)),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(128, activation = 'tanh'),
	tf.keras.layers.Dense(10)
	
])

vid = cv2.VideoCapture(0) 

while(True): 
	
	ret, frame = vid.read() 

	cv2.imshow('frame', frame) 

	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 

cv2.destroyAllWindows() 
