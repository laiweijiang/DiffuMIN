#!/usr/bin/python
# -*- coding:utf-8 -*-

import math

import tensorflow as tf
from tensorflow.python.ops import array_ops

from static import BIG_MODEL_EMBED_TABLE, BIG_MODEL_EMBED_HASH_TABLE


class BaseModel(object):
    def __init__(self, deep_layers=[100, 100], embed_dim=8, learning_rate=0.1, **kwargs):
        self.deep_layers = deep_layers
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

    def __call__(self, features, labels, mode, params):
        raise NotImplementedError

    def inference(self, input_layer, name_scope, activation, out_dim, is_training):
        raise NotImplementedError

    def prelu(self, _x, name):
        alphas = tf.get_variable('alpha' + name, [_x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def dice(self, _x, epsilon=0.000000001, name='', is_training=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)

        pop_mean = tf.get_variable(name='pop_mean' + name, shape=[1, _x.get_shape().as_list()[-1]],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=False)
        pop_std = tf.get_variable(name='pop_std' + name, shape=[1, _x.get_shape().as_list()[-1]],
                                  initializer=tf.constant_initializer(1.0),
                                  trainable=False)

        reduction_axes = 0
        broadcast_shape = [1, _x.shape.as_list()[-1]]
        decay = 0.999
        if is_training:
            mean = tf.reduce_mean(_x, axis=reduction_axes)
            brodcast_mean = tf.reshape(mean, broadcast_shape)
            std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
            std = tf.sqrt(std)
            brodcast_std = tf.reshape(std, broadcast_shape)
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + brodcast_mean * (1 - decay))
            train_std = tf.assign(pop_std,
                                  pop_std * decay + brodcast_std * (1 - decay))
            with tf.control_dependencies([train_mean, train_std]):
                x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
        else:
            x_normed = (_x - pop_mean) / (pop_std + epsilon)
        x_p = tf.sigmoid(x_normed)
        return alphas * (1.0 - x_p) * _x + x_p * _x

    def initialize_hashtable(self,
                             key_dtype,
                             value_dtype,
                             shape,
                             initializer=None,
                             empty_key=-1,
                             shard_num=1,
                             initial_num_buckets=None,
                             shared_name=None,
                             default_value=None,
                             fusion_optimizer_var=False,
                             export_optimizer_var=False,
                             table_impl_type="tbb",
                             name="initialize_hashtable",
                             checkpoint=True,
                             enable_bigmodel=False):
        if enable_bigmodel:
            name = name
            embed_dim = shape.as_list()[-1]
            embed_table = tf.get_variable(
                name,
                shape=[2, embed_dim],
                trainable=False,
                collections=[BIG_MODEL_EMBED_TABLE],
                initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001),
            )

            tf.add_to_collection(BIG_MODEL_EMBED_TABLE, embed_table)

            return embed_table
        else:
            embed_table = tf.contrib.lookup.get_mutable_dense_hashtable(
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                shape=shape,
                initializer=initializer,
                empty_key=empty_key,
                shard_num=shard_num,
                initial_num_buckets=initial_num_buckets,
                shared_name=shared_name,
                default_value=default_value,
                fusion_optimizer_var=fusion_optimizer_var,
                export_optimizer_var=export_optimizer_var,
                table_impl_type=table_impl_type,
                name=name,
                checkpoint=checkpoint
            )

            tf.add_to_collection(BIG_MODEL_EMBED_HASH_TABLE, embed_table)
            return embed_table

    def table_lookup(self, table, is_training, list_ids, threshold, params, v_name, flatten, embedding_size,
                     dump_count_table_to_ckpt=False, model_export=-999):
        model_graph = int(params['model_graph'])
        model_export = int(params['model_export'])
        if model_export < 0:
            model_export = int(params['model_export'])

        def _do_lookup(ids):
            if model_export:
                return self.embedding_lookup_hashtable(
                    emb_tables=table,
                    ids=ids, is_training=is_training,
                    threshold=threshold,
                    dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                    serving_default_value=array_ops.zeros(embedding_size, tf.float32),
                    enable_bigmodel=model_export
                )
            else:
                # 单机导出， 0-正常savedmodel，1-去除表合并的savedmodel
                if model_graph:
                    _embed = self.embedding_lookup_hashtable(
                        emb_tables=table,
                        ids=ids,
                        is_training=is_training,
                        threshold=threshold,
                        dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                        serving_default_value=array_ops.zeros(embedding_size, tf.float32),
                        enable_bigmodel=model_export
                    )
                    return _embed
                else:
                    _unique_ids, _unique_index = tf.unique(ids, out_idx=tf.int64)
                    _unique_embed = self.embedding_lookup_hashtable(
                        emb_tables=table,
                        ids=_unique_ids,
                        is_training=is_training,
                        dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                        threshold=threshold,
                        serving_default_value=array_ops.zeros(
                            embedding_size, tf.float32),
                    )
                    _embed = tf.nn.embedding_lookup(_unique_embed, _unique_index)
                    return _embed

        print("#WSL : list_ids oria is {}".format(list_ids))
        if not isinstance(list_ids, list):
            list_ids = [list_ids]
            print("#WSL : list_ids is {}".format(list_ids))

        len_list = len(list_ids)
        # 单机导出， 0-正常savedmodel，1-去除表合并的savedmodel
        if model_graph:
            list_embed = []
            for ids in list_ids:
                num = ids.shape[1].value
                uniq_ids = tf.reshape(ids, [-1])
                _embed = _do_lookup(uniq_ids)
                if flatten:
                    _embed = tf.reshape(_embed, [-1, num * embedding_size])
                else:
                    _embed = tf.reshape(_embed, [-1, num, embedding_size])
                list_embed.append(_embed)
            message = 'merged table_lookup:\n\t%s' % v_name
            for i in range(len_list):
                message += '\n\t%d\n\t\t%s\n\t\t%s' % (i, list_ids[i], list_embed[i])
            return list_embed

        else:
            list_of_ids = [tf.cast(ids, tf.int64) for ids in list_ids]
            list_of_size = [tf.size(ids) * embedding_size for ids in list_of_ids]
            list_of_num = [ids.shape[1].value for ids in list_of_ids]
            _concat_ids = tf.concat([tf.reshape(ids, [-1]) for ids in list_of_ids], axis=0)
            _embed = _do_lookup(_concat_ids)
            list_embed = tf.split(tf.reshape(_embed, [-1]), list_of_size, axis=0)
            if flatten:
                list_embed = [tf.reshape(list_embed[i], [-1, list_of_num[i] * embedding_size]) for i in range(len_list)]
            else:
                list_embed = [tf.reshape(list_embed[i], [-1, list_of_num[i], embedding_size]) for i in range(len_list)]
            message = 'merged table_lookup:\n\t%s' % v_name
            for i in range(len_list):
                message += '\n\t%d\n\t\t%s\n\t\t%s' % (i, list_ids[i], list_embed[i])
            return list_embed

    def embedding_lookup_hashtable(self,
                                   emb_tables,
                                   ids,
                                   is_training=None,
                                   name=None,
                                   threshold=0,
                                   serving_default_value=None,
                                   enable_bigmodel=False,
                                   dump_count_table_to_ckpt=False):
        if enable_bigmodel:
            emb = tf.nn.embedding_lookup(emb_tables, ids)
            return emb
        else:
            emb = tf.nn.embedding_lookup_hashtable_v2(emb_tables,
                                                      ids,
                                                      is_training=is_training,
                                                      threshold=threshold,
                                                      serving_default_value=serving_default_value,
                                                      dump_count_table_to_ckpt=dump_count_table_to_ckpt)
            return emb
    def get_partitioner(self, partitioner_type=None):
        raise NotImplementedError
