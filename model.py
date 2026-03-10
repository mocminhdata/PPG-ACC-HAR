import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from scipy import signal, stats
from scipy.signal import stft, detrend, butter, filtfilt
from scipy.ndimage import zoom
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
import time
import os
import warnings
warnings.filterwarnings('ignore')


ACTIVITY_LABELS = {
    0: "Transient",
    1: "Sitting",
    2: "Stairs",
    3: "Table soccer",
    4: "Cycling",
    5: "Driving car",
    6: "Lunch break",
    7: "Walking",
    8: "Working",
}

TARGET_FS = 32
SEGMENT_DURATION = 8
SEGMENT_SHIFT = 2
SEGMENT_LENGTH = SEGMENT_DURATION * TARGET_FS
SHIFT_LENGTH = SEGMENT_SHIFT * TARGET_FS
HIGHPASS_CUTOFF = 0.5
FILTER_ORDER = 4

STFT_CONFIG = {
    'PPG': {
        'nperseg': 64, 'hop': 4, 'window': 'hamming',
        'freq_range': (0.5, 4.0), 'use_log': True
    },
    'ACC': {
        'nperseg': 32, 'hop': 2, 'window': 'hann',
        'freq_range': (0, 12), 'use_log': True
    }
}

TARGET_SHAPE = {'freq_bins': 32, 'time_bins': 32, 'channels': 4}

BATCH_SIZE = 32
MAX_EPOCHS = 50
NUM_CLASSES = 8
INPUT_SHAPE = (32, 32, 4)


def LiteSpect_CNN(num_classes=8, input_shape=(32, 32, 4)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(24, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='LiteSpect_CNN')


def compile_model(model):
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        ),
    ]


def compute_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


model = LiteSpect_CNN(NUM_CLASSES, INPUT_SHAPE)
model = compile_model(model)
model.summary()
