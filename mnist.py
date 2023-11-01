import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

data = tf.keras.datasets.fashion_mnist
(training_images,training_labels), (test_images,test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

input_layer = Flatten(input_shape=(28,28))
hidden_layer = Dense(128, activation=tf.nn.relu)
output_layer = Dense(10, activation=tf.nn.softmax)
model = Sequential([input_layer, hidden_layer, output_layer])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
print("About to test model with test data")
model.evaluate(test_images, test_labels)
print("Let's make prediction with the model")
predictions = model.predict(test_images)
print(predictions[0])
print(test_labels[0])