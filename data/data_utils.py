#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def add_fixed_len_column(feature_dict, column_name, shape, tf_type):
    for size in shape:
        if size == 0:
            return
    feature_dict[column_name] = tf.FixedLenFeature(shape, tf_type)


def index_of_tensor(index_map, columns):
    if isinstance(columns, list):
        # index = map(lambda x: index_map[x], columns)
        index = [index_map[i] for i in columns]
    elif isinstance(columns, str):
        index = index_map[columns]
    else:
        raise Exception('columns type is error')
    return index


def extract_list_tensor(raw_list_tensor, index_map, columns):
    feature_list = []
    for col in columns:
        index = index_map[col]
        # field * None * 1
        feature_list.append(raw_list_tensor[:, index])

    # None * field * 1
    feature_list = tf.transpose(feature_list, perm=[1, 0, 2])
    # None * field
    feature_list = tf.reshape(feature_list, [-1, len(columns)])
    return feature_list


def map_list_tensor(raw_list_tensor, index_map, columns):
    index_list = []
    for col in columns:
        index = index_map[col]
        index_list.append(index)
    feature_list = tf.gather(raw_list_tensor, index_list, axis=1)
    feature_list = tf.reshape(feature_list, [-1, len(columns)])
    return feature_list


def extract_tensor_map(raw_list_tensor, index_map, columns):
    feature_map = {}
    for col in columns:
        index = index_map[col]
        # None * 1
        feature_map[col] = raw_list_tensor[:, index]

    return feature_map


dynamic_prefix = 'dynamic_'
dynamic_eff_len_prefix = 'dynamic_eff_len'


def extract_dynamic_tensor_map(ids_list_tensor, feature_list, column_list, max_len_dict):
    index_begin = 0
    dynamic_feature_map = {}
    for feature in column_list:

        dynamic_eff_len_tensor = ids_list_tensor[:, index_begin]
        dynamic_eff_len_tensor = tf.reshape(dynamic_eff_len_tensor, [-1])

        index_begin += 1
        max_len = max_len_dict[feature]
        index_end = index_begin + max_len
        dynamic_tensor = ids_list_tensor[:, index_begin:index_end]
        dynamic_tensor = tf.reshape(dynamic_tensor, [-1, max_len])

        if feature in feature_list:
            dynamic_feature_map[dynamic_prefix + feature] = dynamic_tensor
            dynamic_feature_map[dynamic_eff_len_prefix + feature] = dynamic_eff_len_tensor

        index_begin = index_end

    return dynamic_feature_map


def extract_dynamic_tensor_list(ids_list_tensor, feature_list, column_list, max_len_dict):
    index_begin = 0
    dynamic_feature_map = {}
    for feature in column_list:
        index_begin += 1
        max_len = max_len_dict[feature]
        index_end = index_begin + max_len
        dynamic_tensor = ids_list_tensor[:, index_begin:index_end]
        dynamic_tensor = tf.reshape(dynamic_tensor, [-1, max_len])

        if feature in feature_list:
            dynamic_feature_map[feature] = dynamic_tensor

        index_begin = index_end
    return dynamic_feature_map


def get_list_from_map(total_map, key_list):
    res_map = {}
    for key in key_list:
        res_map[key] = total_map[key]
    return res_map


def get_dynamic_from_map(total_map, key_list):
    res_map = {}
    for key in key_list:
        ids_tensor_key = dynamic_prefix + key
        len_tensor_key = dynamic_eff_len_prefix + key

        ids_tensor = total_map[ids_tensor_key]
        len_tensor = total_map[len_tensor_key]

        res_map[key] = (ids_tensor, len_tensor)
    return res_map


def parse_fn(serialized_example, data_struct):
    tf_type_map = {'int': tf.int64, 'long': tf.int64, 'str': tf.string, 'float': tf.float32,
                   'string': tf.string, 'double': tf.float32}
    feature = {
        'label': tf.FixedLenFeature([1], tf.float32)
    }
    for name, column in data_struct.columns_dict.iteritems():
        column_size = len(column.columns)
        tf_type = tf_type_map[column.type]
        add_fixed_len_column(feature, name, [column_size], tf_type)

    features = tf.parse_example(serialized_example, features=feature)

    return features, features['label']


def input_fn(file_names, epoch, batch_size, parse_func, prefetch_num):
    if file_names is None:
        dataset = tf.data.AfoDataset()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(parse_func, num_parallel_calls=16)
        dataset = dataset.prefetch(prefetch_num)
    else:
        files = tf.data.Dataset.list_files(file_names, shuffle=False)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=8))
        dataset = dataset.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_func, num_parallel_calls=8)
        dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()




def extract_labels(label_tensor):
    label_result = []
    with tf.Session() as sess:
        while True:
            try:
                labels = sess.run(label_tensor)
                label_result.extend(labels)
            except Exception, e:
                break

    return label_result


def get_embed_table_index_base(feat_size_dict, feat_list):
    index_base = []
    base = 0
    for feat in feat_list:
        index_base.append(base)
        feat_size = feat_size_dict[feat]
        base += feat_size
    return index_base


def get_total_embed_table_size(feat_size_dict, feat_list):
    size_list = [feat_size_dict[feat] for feat in feat_list]
    total_size = sum(size_list)
    return total_size


def get_checkpoint_list(model_path):
    name_set = set()
    all_files = tf.gfile.ListDirectory(model_path)
    for name in all_files:
        if name.startswith('model.ckpt-'):
            parts = name.split('.')
            if len(parts) < 3:
                continue
            name_prefix = '.'.join(parts[0:2])
            index = int(name_prefix.split('-')[1])
            name_set.add((name_prefix, index))
    res_list = list(name_set)
    res_list = [item for item in res_list if item[1] > 0]
    res_list = sorted(res_list, key=lambda v: v[1])

    res_list = [item[0] for item in res_list]
    return res_list