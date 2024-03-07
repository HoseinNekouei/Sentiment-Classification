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


MAX_TOKEN =  10000
MAX_LEN = 512
EPOCHS = 10
BATCH_SIZE = 64

def load_text_dataset(path_info: str) -> tuple[np.array, np.array]:
    '''
        Load a CSV file from the specified directory.

        Args:
            directory_path (str):
                Path to the directory containing the text data.

            batch_size(int): 
                Number of samples per batch.

        Return:
            features (array): The features data.
            labels (array): The labels data.
    '''

    # Read cvs file in memory
    data = pd.read_csv('amazon_baby.csv', usecols=[1,2], nrows= 10000)
    print(f'shape of dataset befor decrease majority classes: {data.shape}')

    # drop null reviews
    empty_idxs = data[data['review'].isnull()].index
    data.drop(empty_idxs, inplace=True)

    ## drop some samples of the majority classes randomly
    # mj_classes = data[data['rating'] > 2]
    # mj_index = mj_classes.sample( frac = 0.4, replace = False).index
    # data.drop(mj_index, inplace= True)
    
    # drop 40 percent of samples that represent the majority classes randomly.
    mj_idx = data.sample(frac = 0.4 , weights= data['rating']).index
    data.drop(mj_idx)

    print(f'shape of dataset after decrease majorite classes: {data.shape}')

    # Shuffle the dataframe rows
    data = data.sample(frac=1)

    # Split labels and features. Change labels > 2 to 0, others to 1
    features = data['review'].astype(str)
    labels = data['rating'].apply(lambda x: 0 if x > 0 and x <=2 else 1)

    # Print the shuffled dataframe for the top 100 records 
    print(labels[:100].tolist())

    return (features, labels)


def preprocess_text_data(features, labels):

    '''
        get features and labels and preprocesses text data from the specified datasets.

        Args:
            Features:
                Features dataset containing text samples.

            Labels:
                Labels that containing true integer samples.

            Max_token (int): 
                Maximum number of tokens in the vocabulary.

            Output_mode (str):
                Output mode ('int' for integer-encoded tokens).

            Output_sequence_length (int):
                Maximum sequence length for the output (length of the dictionary).

        Returns:
            x_train (array): 
                The training data.

            x_test (array): 
                The testing data.

            y_train (array): 
                The training target.

            y_test (array): 
                The testing target.

            class_weights (dict): 
                A dictionary mapping class indices to their respective weights.
    '''

    #compute class weights when taking into account the distribution of labels
    class_weights = compute_class_weight('balanced', classes= np.unique(labels), y= labels )
    class_weights= {i: class_weights[i] for i in range(2)}
    print(f'weight classes: {class_weights}')

    # Split dataset into training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels, 
                                                        test_size=0.1, 
                                                        random_state= 42)
    print(f'x_train.shape: {x_train.shape}')
    print(f'x_test.shape: {x_test.shape}')
    print(f'y_tain.shape: {y_train.shape}')
    print(f'y_test.shape: {y_test.shape}')
    print(x_train.iloc[0])

    # Create a TextVectorization layer
    vectorizer = layers.TextVectorization(max_tokens= MAX_TOKEN,
                                                    output_mode= 'int',
                                                    output_sequence_length= MAX_LEN)

    # Adapt the TextVectorization layer to text data
    vectorizer.adapt(x_train, batch_size= BATCH_SIZE)

    print(vectorizer.get_vocabulary()[:10])    

    return (vectorizer, class_weights, x_train, x_test, y_train, y_test)


def algorithm(vectorizer, class_weights, x_train, x_test, y_train, y_test):
    '''
        Create an LSTM_based binary classification model.

        Args:
            max_token (int):
                Maximun number of tokens in the vocabulary.
            
        returns:
            history (keras.callbacks.History): 
                A History object containing training/validation loss and metrics.
    '''

    # Define input layer
    inputs = keras.Input(shape = (1,), dtype = tf.string)

    # vectorize input
    vectorize = vectorizer(inputs)

    # Embedding layer
    embeded= layers.Embedding(input_dim= MAX_TOKEN, output_dim= 256)(vectorize)

    # create LSTM model
    x = layers.LSTM(units= 64, recurrent_dropout=0.5, return_sequences=True)(embeded)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units= 32, recurrent_dropout=0.5, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units= 32, recurrent_dropout=0.5)(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create Model
    model = Model(inputs, outputs)

    # Define optimizer with custom learning rate
    opt =  keras.optimizers.Adam(learning_rate= 0.001)

    # Compile the model
    model.compile(optimizer= opt,
                loss = 'binary_crossentropy',
                metrics= 'accuracy')

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

    return history


def show_results(history):
    
    """
    Plot the training and validation loss/accuracy from the history object.

    Args:
        history (keras.callbacks.History): 
            The history object returned from model training.
        
        EPOCHS (int): The number of epochs for which the model was trained.

    Returns:
        None
    """
    
    # Set the plot style to 'ggplot'
    plt.style.use('ggplot')

    # Extract loss and accuracy values from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Plot the training and validation loss
    plt.plot(np.arange(EPOCHS), loss, label='Loss')
    plt.plot(np.arange(EPOCHS), val_loss, label= 'Val_loss')

    # Plot the training and validation accuracy
    plt.plot(np.arange(EPOCHS), accuracy, label='accuracy')
    plt.plot(np.arange(EPOCHS), val_accuracy, label= 'Val_accuracy')

    # Add legend, labels, and title to the plot
    plt.legend()
    plt.xlabel('Epotch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Amazon sentiment analysis')

    # Display the plot
    plt.show()


def main():
    print('The main function is runnig ...')

    feature, labels = load_text_dataset(r'amazon_baby.csv')
    vectorizer, class_weights, x_train, x_test, y_train, y_test  = preprocess_text_data(feature, labels)
    history = algorithm(vectorizer, class_weights, x_train, x_test, y_train, y_test)
    show_results(history)


if __name__== '__main__': 
    main()
