import sklearn.metrics
import pandas as pd
import numpy as np
import keras.utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, BatchNormalization, Dropout, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.optimizers
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
import tensorflow as tf
import random
from keras.layers import Bidirectional
import spacy

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return (1 - jac)


#random.seed(42)
#np.random.seed(42)
#tf.set_random_seed(42)

def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    # Hyperparameters to tune
    max_words = 50
    batch_size = 64
    epchs = 10
    CategoricalThreshold = 0.3
    ################## NO Changes, some initializations #######################
    accu = np.zeros((epchs, 2))
    n_out = len(emotions)
    dev_predictions = dev_data.copy()
    nlp = spacy.load('en')
    test_predictions = test_data.copy()
    
    ################## Preprocessing Language Data#############################
    
    # Pre-processing for Train Data
    trainingData = train_data['Tweet'].values
    tokenizer1 = Tokenizer()
    tokenizer1.fit_on_texts(trainingData)
    X = tokenizer1.texts_to_sequences(trainingData)
    convertedData = tokenizer1.sequences_to_texts(X)
    
    finalTrain = list()
    for i in range(0, tokenizer1.document_count):
        seq = list()
        for word in nlp(convertedData[i]):
            seq.append(word.lemma_)
        line = ' '.join(seq)
        finalTrain.append(line)
    
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(finalTrain)
    X = tokenizer2.texts_to_sequences(finalTrain)
    X = pad_sequences(X, maxlen = max_words, padding = 'post', value = 0)
    Y = train_data[emotions].values
    
    # Pre-processing for Dev Data
    tokenizer3 = Tokenizer()
    tokenizer3.fit_on_texts(dev_data['Tweet'].values)
    XDev = tokenizer3.texts_to_sequences(dev_data['Tweet'].values)
    convertedDevData = tokenizer3.sequences_to_texts(XDev)
    
    finalDev = list()
    for i in range(0, tokenizer3.document_count):
        seq = list()
        for word in nlp(convertedDevData[i]):
            seq.append(word.lemma_)
        line = ' '.join(seq)
        finalDev.append(line)
    
    XDev = tokenizer2.texts_to_sequences(finalDev)
    XDev = pad_sequences(XDev, maxlen = max_words, padding = 'post', value = 0)
    YDev = dev_data[emotions].values
    
    # Pre-processing for Test Data
    tokenizer4 = Tokenizer()
    tokenizer4.fit_on_texts(test_data['Tweet'].values)
    XTest = tokenizer4.texts_to_sequences(test_data['Tweet'].values)
    convertedTestData = tokenizer4.sequences_to_texts(XTest)
    
    finalTest = list()
    for i in range(0, tokenizer4.document_count):
        seq = list()
        for word in nlp(convertedTestData[i]):
            seq.append(word.lemma_)
        line = ' '.join(seq)
        finalTest.append(line)
    
    XTest = tokenizer2.texts_to_sequences(finalTest)
    XTest = pad_sequences(XTest, maxlen = max_words, padding = 'post', value = 0)
    
    ################## Custom Callback for jaccard score ######################
    class SensitivitySpecificityCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            x_test = self.validation_data[0]
            y_test = self.validation_data[1]
            predictions = np.where(self.model.predict(x_test)<CategoricalThreshold, 0, 1)
            accu[epoch, 1] = sklearn.metrics.jaccard_similarity_score(y_test, predictions)
            print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
                    y_test, predictions)))
    
    ################## Main Model #############################################
    model = Sequential()
    model.add(Embedding(len(tokenizer2.word_counts) + 1, 150 , input_length = X.shape[1], trainable = True, mask_zero=False))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU((128), activation = 'tanh', return_sequences = True, dropout = 0.2)))
    model.add((GRU((256), activation = 'tanh', return_sequences = True, dropout = 0.8)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_out, activation = 'sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics = ["accuracy"])
    model.compile(loss=[jaccard_distance], optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics = ["accuracy"])
    
    ################## Printing and plotting model ############################
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB')
    history = model.fit(x = X,y = Y, epochs = epchs, batch_size=batch_size, verbose = 2, validation_data=(XDev, YDev), callbacks=[SensitivitySpecificityCallback(), EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=2)])#restore_best_weights=True, patience=2)])
    
    # Plotting script
    
    # Loss Plot
    plt.plot(history.history['loss'], color = 'b', label='loss')
    plt.plot(history.history['val_loss'], color = 'g', label='val_loss')
    plt.show()
    
    # Accuracy Plot
    plt.plot(history.history['acc'], color = 'b', label='categorical_accuracy')
    plt.plot(history.history['val_acc'], color = 'g', label='val_categorical_accuracy')
    plt.show()
    
    # Dev Data Jaccard Similarity Plot
    plt.plot(accu[:,1], color = 'g', label='val_binary_accuracy')
    plt.show()
    
    test_predictions[emotions] = np.where(model.predict(XTest)<CategoricalThreshold, 0, 1)
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    dev_predictions[emotions] = np.where(model.predict(XDev)<CategoricalThreshold, 0, 1)
    return dev_predictions
    
if __name__ == "__main__":

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv("2018-E-c-En-train.txt", **read_csv_kwargs)
    dev_data = pd.read_csv("2018-E-c-En-dev.txt", **read_csv_kwargs)
    test_data = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)
    # makes predictions on the dev set
    dev_predictions = train_and_predict(train_data, dev_data, test_data)

    # saves predictions and prints out multi-label accuracy
    #dev_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    print("dev_accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        dev_data[emotions], dev_predictions[emotions])))
