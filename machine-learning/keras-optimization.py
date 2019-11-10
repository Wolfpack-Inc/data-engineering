# Tensorflow imports
import os
import logging
from math import sqrt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model, model_from_json, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, quniform

# Sklearn imports
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Mongo
from pymongo import MongoClient

# Pandas and numpy
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# Other
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    def split_sequence(sequence, n_steps, y_col=0):
        """
        Function to create historic sequences
        """
        X, y = [], []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X)

    # Mongo client
    client = MongoClient('mongodb://165.22.199.122:27017')
    db = client.processed
    data = db.external

    # Load the data and set the timestamp as index
    df = (pd.DataFrame(list(data.find()))
        .drop('_id', 1)
        .set_index('timestamp')
        .sort_index()
        .dropna())

    df['price_diff'] = df['price'].diff()
    df = df.dropna()

    n_steps = 24
    train_size = int(.75 * len(df))

    # Train test split
    prices = df['price'].values
    prices_train, prices_test = prices[:train_size], prices[train_size:]

    sentiment = df['sentiment'].values
    sentiment_train, sentiment_test = sentiment[:train_size], sentiment[train_size:]

    n_tweets = df['n_tweets'].values
    n_tweets_train, n_tweets_test = n_tweets[:train_size], n_tweets[train_size:]

    train = np.stack([prices_train, sentiment_train, n_tweets_train], axis=1)
    test = np.stack([prices_test, sentiment_test, n_tweets_test], axis=1)

    # Scale the data
    scaler = Normalizer()

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Generate sequences
    x_train = split_sequence(train_scaled, n_steps)
    x_test = split_sequence(test_scaled, n_steps)

    # y_prices = df['price'].diff()
    y_prices = df['price']

    y_test = y_prices.iloc[-len(x_test):].values
    y_train = y_prices.iloc[n_steps:len(x_train)+n_steps].values

    # Flatten the outputs (Input on dense layers)
    x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2])

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    n_steps = 24
    n_features = 3

    model = Sequential()
    model.add(Dense(int({{quniform(4,64,1)}}), activation='relu', input_shape=(n_steps*n_features,)))
    model.add(Dropout(rate={{uniform(0,1)}}))
    
    for _ in range(int({{quniform(1,3,1)}})):
        model.add(Dense(units=int({{quniform(4,64,1)}}), activation='relu'))
        model.add(Dropout(rate={{uniform(0, 1)}}))
        
    model.add(Dense(1))
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss=root_mean_squared_error)

    early_stopping = EarlyStopping(monitor='val_loss', patience=24)
    checkpointer = ModelCheckpoint(filepath='keras_weights_{}.hdf5'.format(model.name),
                                   verbose=1,
                                   save_best_only=True)

    result = model.fit(x_train, y_train,
                       batch_size=int({{quniform(64,512,1)}}),
                       epochs=int({{quniform(64,512,1)}}),
                       verbose=2,
                       validation_split=0.2,
                       callbacks=[early_stopping, checkpointer])

    validation_loss = np.amin(result.history['val_loss'])
    print('Best validation loss of epoch:', validation_loss)
    print(model.summary())

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials(),
                                          eval_space=True)

    x_train, y_train, x_test, y_test = data()

    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    print(best_model.summary())

    best_model.save('best_model.h5')
