from __future__ import print_function
from load_data import load_data_shuffle
from keras import backend as K
from keras.layers import Dense, Dropout, Lambda, Embedding, Conv1D, LSTM, Input
from keras.layers import concatenate
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from keras.utils import pad_sequences
import numpy as np
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Thiết lập các thông số:
np.random.seed(1337)  # để tái tạo kết quả
max_features = 21540  # 14300
maxlen = 400
batch_size = 10
embedding_dims = 200
nb_filter = 150
filter_length = 3
hidden_dims = 100
nb_epoch = 14
cvs = [1, 2, 3, 4, 5]
accs = []
for cv in cvs:
    print(f'Đang tải dữ liệu cho cv...{cv}')

    X_train, y_train, X_test, y_test, X_val, y_val = load_data_shuffle(cv)
    print(f'Độ dài chuỗi dữ liệu huấn luyện: {len(X_train)}')
    print(f'Độ dài chuỗi dữ liệu kiểm tra: {len(X_test)}')

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)
    print(f'Kích thước X_train: {X_train.shape}')
    print(f'Kích thước X_val: {X_val.shape}')
    print(f'Kích thước X_test: {X_test.shape}')

    print('Đang xây dựng mô hình...')
    model = Sequential()

    input_layer = Input(shape=(maxlen,), dtype='int32', name='main_input')
    emb_layer = Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen
                            )(input_layer)

    def max_1d(X):
        return K.max(X, axis=1)

    # Các tầng tích chập

    con3_layer = Conv1D(filters=nb_filter,
                        kernel_size=3,
                        padding='valid',
                        activation='relu')(emb_layer)

    pool_con3_layer = Lambda(max_1d, output_shape=(nb_filter,))(con3_layer)

    con4_layer = Conv1D(filters=nb_filter,
                        kernel_size=5,
                        padding='valid',
                        activation='relu')(emb_layer)

    pool_con4_layer = Lambda(max_1d, output_shape=(nb_filter,))(con4_layer)

    con5_layer = Conv1D(filters=nb_filter,
                        kernel_size=7,
                        padding='valid',
                        activation='relu')(emb_layer)

    pool_con5_layer = Lambda(max_1d, output_shape=(nb_filter,))(con5_layer)

    cnn_layer = concatenate([pool_con3_layer, pool_con5_layer,
                            pool_con4_layer])

    # Tầng LSTM

    x = Embedding(max_features, embedding_dims,
                    input_length=maxlen)(input_layer)
    lstm_layer = LSTM(128)(x)

    cnn_lstm_layer = concatenate([lstm_layer, cnn_layer])

    dense_layer = Dense(
        hidden_dims*2, activation='sigmoid')(cnn_lstm_layer)
    output_layer = Dropout(0.2)(dense_layer)
    output_layer = Dense(
        3, trainable=True, activation='softmax')(output_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

    model.compile(loss='categorical_crossentropy',
                    optimizer="adamax",
                    metrics=['accuracy'])
    checkpoint = ModelCheckpoint('CNN-LSTM-weights/cv'+str(cv)+'weights.hdf5',
                                    monitor='val_accuracy', verbose=2,
                                    mode='max', save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                callbacks=[checkpoint],
                validation_data=(X_val, y_val))

    model.load_weights('CNN-LSTM-weights/cv'+str(cv)+'weights.hdf5')
    model.compile(loss='categorical_crossentropy',
                    optimizer="adamax",
                    metrics=['accuracy'])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("Độ chính xác: ", acc)
    accs.append(acc)
print("Độ chính xác trung bình: ", K.np.mean(K.np.array(accs)))
