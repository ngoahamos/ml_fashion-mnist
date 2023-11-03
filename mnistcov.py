import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from mycallback import MyCallback


data = tf.keras.datasets.fashion_mnist
(training_images,training_labels), (test_images,test_labels) = data.load_data()

print("Number of training images {}".format(len(training_images)))
print("Number of test images {}".format(len(test_images))) 
training_images = training_images.reshape(len(training_images),28,28,1)
training_images = training_images / 255.0

test_images = test_images.reshape(len(test_images), 28,28,1)
test_images = test_images / 255.0

initial_cov = Conv2D(64,(3,3), activation=tf.nn.relu, input_shape=(28,28,1))
max_pooling = MaxPooling2D(2,2)
second_cov = Conv2D(64,(3,3), activation=tf.nn.relu)

input_layer = Flatten()
hidden_layer = Dense(128, activation=tf.nn.relu)
output_layer = Dense(10, activation=tf.nn.softmax)

model = Sequential([initial_cov,max_pooling, second_cov, max_pooling, 
                    input_layer,hidden_layer, output_layer])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback = MyCallback()

model.fit(training_images, training_labels, epochs=50, callbacks = [callback]) #callbacks = [callback]
print("About to test model with test data")
model.evaluate(test_images, test_labels)
print("Let's make prediction with the model")
predictions = model.predict(test_images)
print("Model Prediction")
print(predictions[0])
print(test_labels[0])