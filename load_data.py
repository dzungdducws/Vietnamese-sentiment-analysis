import numpy as np
import os
import pickle


def load_data_shuffle(cv):
    train_pos_save = f"data/data_token/fold_{cv}/train_pos.npy"
    train_pos_save = f"data/data_token/fold_{cv}/train_neg.npy"
    train_pos_save = f"data/data_token/fold_{cv}/train_neu.npy"

    test_pos_save = f"data/data_token/fold_{cv}/test_pos.npy"
    test_neg_save = f"data/data_token/fold_{cv}/test_neg.npy"
    test_neu_save = f"data/data_token/fold_{cv}/test_neu.npy"

    # Load dữ liệu train
    pos_train = np.load(train_pos_save, encoding='bytes', allow_pickle=True)
    neg_train = np.load(train_pos_save, encoding='bytes', allow_pickle=True)
    neu_train = np.load(train_pos_save, encoding='bytes', allow_pickle=True)

    y_pos_train = np.array([[1, 0, 0]] * len(pos_train))
    y_neg_train = np.array([[0, 1, 0]] * len(neg_train))
    y_neu_train = np.array([[0, 0, 1]] * len(neu_train))

    # load dữ liệu test
    pos_test = np.load(test_pos_save, encoding='bytes', allow_pickle=True)
    neg_test = np.load(test_neg_save, encoding='bytes', allow_pickle=True)
    neu_test = np.load(test_neu_save, encoding='bytes', allow_pickle=True)

    y_pos_test = np.array([[1, 0, 0]] * len(pos_test))
    y_neg_test = np.array([[0, 1, 0]] * len(neg_test))
    y_neu_test = np.array([[0, 0, 1]] * len(neu_test))

    # Split train and validate set
    val_len = len(pos_train) // 10

    pos_val = pos_train[:val_len]
    pos_train = pos_train[val_len:]
    y_pos_val = y_pos_train[:val_len]
    y_pos_train = y_pos_train[val_len:]

    neg_val = neg_train[:val_len]
    neg_train = neg_train[val_len:]
    y_neg_val = y_neg_train[:val_len]
    y_neg_train = y_neg_train[val_len:]

    neu_val = neu_train[:val_len]
    neu_train = neu_train[val_len:]
    y_neu_val = y_neu_train[:val_len]
    y_neu_train = y_neu_train[val_len:]

    X_train = np.concatenate([pos_train, neu_train, neg_train])
    y_train = np.concatenate([y_pos_train, y_neu_train, y_neg_train])

    X_val = np.concatenate([pos_val, neu_val, neg_val])
    y_val = np.concatenate([y_pos_val, y_neu_val, y_neg_val])

    X_test = np.concatenate([pos_test, neu_test, neg_test])
    y_test = np.concatenate([y_pos_test, y_neu_test, y_neg_test])

    # print(f"X_train: {X_train}")
    # print(f"y_train: {y_train}")
    # print(f"X_val: {X_val}")
    # print(f"y_val: {y_val}")
    # print(f"X_test: {X_test}")
    # print(f"y_test: {y_test}")

    return X_train, y_train, X_test, y_test, X_val, y_val
