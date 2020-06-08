#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import warnings

import mlflow
import mlflow.tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings('ignore')


# In[ ]:


DATA_DIR_PATH = './data'
CLOSED_DATA_FILENAME = 'olist_closed_deals_dataset.csv'
QUALIFIED_DATA_FILENAME = 'olist_marketing_qualified_leads_dataset.csv'

RESOURCES_DIR_PATH = './resources'
LANDING_PAGE_ID_VOCABULARY_FILENAME = 'landing_page_id_vocabulary.txt'
ORIGIN_VOCABULARY_FILENAME = 'origin.txt'
PROCESSED_TRAIN_DATA_FILENAME = 'processed_train_data.csv'
PROCESSED_TEST_DATA_FILENAME = 'processed_test_data.csv'

SPLIT_RATIO = 0.8
BATCH_SIZE = 1024

EPOCHS = 100
STEPS_PER_EPOCH = 20

DEFAULT_NAN_STRING = 'missing'
DEFAULT_NAN_INT = -1

MODEL_DIR = './models'
MODEL_NAME = 'olist_model'


# In[ ]:


def validate_data_paths():
    data_dir = pathlib.Path(DATA_DIR_PATH)
    assert data_dir.exists() and data_dir.is_dir()

    closed_deals_path = data_dir / CLOSED_DATA_FILENAME
    assert closed_deals_path.exists() and closed_deals_path.is_file()

    qualified_deals_path = data_dir / QUALIFIED_DATA_FILENAME
    assert qualified_deals_path.exists() and qualified_deals_path.is_file()


# In[ ]:


def preprocess():
    data_dir = pathlib.Path(DATA_DIR_PATH)

    closed_deals_path = data_dir / CLOSED_DATA_FILENAME
    closed_df = pd.read_csv(closed_deals_path)

    qualified_deals_path = data_dir / QUALIFIED_DATA_FILENAME
    qualified_df = pd.read_csv(qualified_deals_path)

    raw_df = qualified_df.merge(closed_df, on='mql_id', how='left')

    filtered_df = raw_df[['landing_page_id', 'origin']]
    filtered_df['origin'] = filtered_df['origin'].fillna(DEFAULT_NAN_STRING)
    filtered_df['label'] = raw_df['seller_id'].notna()

    resources_dir = pathlib.Path(RESOURCES_DIR_PATH)
    if not resources_dir.exists():
        resources_dir.mkdir()

    landing_page_id_vocabulary_file = resources_dir / \
        LANDING_PAGE_ID_VOCABULARY_FILENAME
    if landing_page_id_vocabulary_file.exists():
        landing_page_id_vocabulary_file.unlink()

    origin_vocabulary_file = resources_dir / ORIGIN_VOCABULARY_FILENAME
    if origin_vocabulary_file.exists():
        origin_vocabulary_file.unlink()

    np.savetxt(landing_page_id_vocabulary_file,
               pd.unique(filtered_df['landing_page_id']), fmt='%s')
    np.savetxt(origin_vocabulary_file,
               pd.unique(filtered_df['origin']), fmt='%s')

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
    split = int(len(filtered_df) * SPLIT_RATIO)

    train_df = filtered_df[:split]
    test_df = filtered_df[split:]

    processed_train_data_file = resources_dir / PROCESSED_TRAIN_DATA_FILENAME
    if processed_train_data_file.exists():
        processed_train_data_file.unlink()

    processed_test_data_file = resources_dir / PROCESSED_TEST_DATA_FILENAME
    if processed_test_data_file.exists():
        processed_test_data_file.unlink()

    train_df.to_csv(processed_train_data_file, index=False, header=True)
    test_df.to_csv(processed_test_data_file, index=False, header=True)


# In[ ]:


def validate_preprocess_step():
    resources_dir = pathlib.Path(RESOURCES_DIR_PATH)
    assert resources_dir.exists()

    landing_page_id_vocabulary_file = resources_dir / \
        LANDING_PAGE_ID_VOCABULARY_FILENAME
    assert landing_page_id_vocabulary_file.exists()

    origin_vocabulary_file = resources_dir / ORIGIN_VOCABULARY_FILENAME
    assert origin_vocabulary_file.exists()

    processed_train_data_file = resources_dir / PROCESSED_TRAIN_DATA_FILENAME
    assert processed_train_data_file.exists()

    processed_test_data_file = resources_dir / PROCESSED_TEST_DATA_FILENAME
    assert processed_test_data_file.exists()


# In[ ]:


def load_data():
    resources_dir = pathlib.Path(RESOURCES_DIR_PATH)
    processed_train_data_file = resources_dir / PROCESSED_TRAIN_DATA_FILENAME
    processed_test_data_file = resources_dir / PROCESSED_TEST_DATA_FILENAME

    train_df = pd.read_csv(processed_train_data_file)
    test_df = pd.read_csv(processed_test_data_file)

    train_x, train_y = train_df, train_df.pop('label')
    test_x, test_y = test_df, test_df.pop('label')

    return (train_x, train_y), (test_x, test_y)


# In[ ]:


def get_dataset(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(len(features)).repeat().batch(batch_size)

    return dataset


# In[ ]:


def get_stats_from_data(labels):
    total = len(labels)
    negative, positive = np.bincount(labels)

    weight_for_0 = (1 / negative) * (total) / 2.0
    weight_for_1 = (1 / positive) * (total) / 2.0

    class_weight = {False: weight_for_0, True: weight_for_1}

    output_bias = np.log([positive / negative])

    return class_weight, output_bias


# In[ ]:


def get_model(output_bias=None):
    resources_dir = pathlib.Path(RESOURCES_DIR_PATH)
    landing_page_id_vocabulary_file = resources_dir / \
        LANDING_PAGE_ID_VOCABULARY_FILENAME
    origin_vocabulary_file = resources_dir / ORIGIN_VOCABULARY_FILENAME

    landing_page_id_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='landing_page_id',
        vocabulary_file=str(landing_page_id_vocabulary_file),
        dtype=tf.dtypes.string,
        num_oov_buckets=1
    )

    origin_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='origin',
        vocabulary_file=str(origin_vocabulary_file),
        dtype=tf.dtypes.string,
        num_oov_buckets=1
    )

    feature_columns = [tf.feature_column.indicator_column(landing_page_id_column),
                       tf.feature_column.indicator_column(origin_column)]

    feature_layer_inputs = {}
    feature_layer_inputs['landing_page_id'] = tf.keras.Input(
        shape=(1,), name='landing_page_id', dtype=tf.string)
    feature_layer_inputs['origin'] = tf.keras.Input(
        shape=(1,), name='origin', dtype=tf.string)

    if output_bias:
        output_bias = tf.keras.initializers.Constant(output_bias)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    dense = feature_layer(feature_layer_inputs)

    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(10, activation='relu')(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(10, activation='relu')(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    output = tf.keras.layers.Dense(
        1, activation='sigmoid', bias_initializer=output_bias)(dense)

    model = tf.keras.Model(
        inputs=[v for v in feature_layer_inputs.values()], outputs=output)

    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )

    return model


# In[ ]:


def train():
    (train_x, train_y), (test_x, test_y) = load_data()
    train_dataset = get_dataset(train_x, train_y, BATCH_SIZE)
    test_dataset = get_dataset(test_x, test_y, BATCH_SIZE)

    class_weight, output_bias = get_stats_from_data(train_y)

    model = get_model(output_bias)

    history = model.fit(train_dataset, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH, class_weight=class_weight, verbose=0)

    model_dir = pathlib.Path(MODEL_DIR)
    if not model_dir.exists():
        model_dir.mkdir()

    model.save(model_dir / MODEL_NAME)


# In[ ]:


def validate_train():
    model_dir = pathlib.Path(MODEL_DIR)
    assert model_dir.exists()

    saved_model_dir = model_dir / MODEL_NAME
    assert saved_model_dir.exists()

    assert (saved_model_dir / 'assets').exists()
    assert (saved_model_dir / 'variables').exists()
    assert (saved_model_dir / 'saved_model.pb').exists()


# In[ ]:


def main():
    validate_data_paths()
    preprocess()
    validate_preprocess_step()

    mlflow.tensorflow.autolog()

    with mlflow.start_run() as run:
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.keras", tf.keras.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)

        train()
        validate_train()


# In[ ]:


main()
