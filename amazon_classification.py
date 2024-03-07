import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Model, layers
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


MAX_TOKEN =  7500
MAX_LEN = 256
EPOCHS = 10
BATCH_SIZE = 32

# Read cvs file in memory
data = pd.read_csv('amazon_baby.csv', usecols=[1,2], nrows=90000)

# drop null reviews
empty_idxs = data[data['review'].isnull()].index
data.drop(empty_idxs, inplace=True)

data = data.sample(frac=1)
features = data['review'].astype(str)
labels = data['rating'].apply(lambda x: 0 if x > 0 and x <=2 else 1)

print(labels[:100].tolist())

#compute class weights
# The weight of the classes when taking into account the distribution of labels
class_weights = compute_class_weight('balanced', classes= np.unique(labels), y= labels )
class_weights= {i: class_weights[i] for i in range(2)}
print(f'Labels weight: {class_weights}')

# Split dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    labels, 
                                                    test_size=0.1, 
                                                    random_state= 42)
print(x_train.shape)

# Create a TextVectorization layer
vectorizer = layers.TextVectorization(max_tokens= MAX_TOKEN,
                                                output_mode= 'int',
                                                output_sequence_length= MAX_LEN)

# Adapt the TextVectorization layer to text data
vectorizer.adapt(x_train)

print(vectorizer.get_vocabulary()[:10])

# Define input layer
inputs = keras.Input(shape = (1,), dtype = tf.string)

# vectorize input
vectorize = vectorizer(inputs)

# Embedding layer
embeded= layers.Embedding(input_dim= MAX_TOKEN, output_dim= 256)(vectorize)

# create LSTM model
x = layers.LSTM(units= 64, recurrent_dropout=0.5, return_sequences=True)(embeded)
x = layers.LSTM(units= 32, recurrent_dropout=0.5, return_sequences=True)(x)
x = layers.LSTM(units= 32, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)

# Output layer
outputs = layers.Dense(1, activation='sigmoid')(x)

# Create Model
model = Model(inputs, outputs)

# Define optimizer with custom learning rate
opt =  keras.optimizers.Adam(learning_rate= 0.1)

# Compile the model
model.compile(optimizer= opt,
              loss = 'binary_crossentropy',
              metrics= 'accuracy',
              )

model.summary()

history = model.fit(x_train, 
                    y_train, 
                    validation_data= (x_test, y_test), 
                    epochs = EPOCHS, 
                    batch_size= BATCH_SIZE,
                    class_weight= class_weights)

loss, accuracy = model.evaluate(x_test, y_test) 
print(f'Test loss: {loss :.2f}, Test accuracy: {accuracy :.2f}')

# save model
model.save("amazon_model", save_format='tf')

plt.style.use('ggplot')

# Get the loss and accuracy values from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(np.arange(EPOCHS), loss, label='Loss')
plt.plot(np.arange(EPOCHS), val_loss, label= 'Val_loss')

plt.plot(np.arange(EPOCHS), accuracy, label='accuracy')
plt.plot(np.arange(EPOCHS), val_accuracy, label= 'Val_accuracy')

plt.legend()
plt.xlabel('Epotch')
plt.ylabel('Loss/Accuracy')
plt.title('Amazon sentiment analysis')

plt.show()