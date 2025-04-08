#!/usr/bin/python
# -*- coding:utf-8 -*-
import functools

import tensorflow as tf
import math
from model_lib.base_model import BaseModel
from utils.utils import import_class
import utils.CONST as cst
import utils as ut
from data.data_utils import index_of_tensor
import itertools as it
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *

class DNN(BaseModel):
    def __init__(self, data_struct, params, logger, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.data_struct = data_struct
        self.params = params
        self.logger = logger

        self.deep_layers = params['deep_layers']
        self.embed_dim = params['embed_dim']
        self.learning_rate = params['learning_rate']
        self.ps_num = params['ps_num']

        self._partitioner_set = {'fix', 'var', 'min_max'}
        _partitioner = self.params.get('partitioner', 'fix')
        self._partitioner = _partitioner if _partitioner in self._partitioner_set else 'fix'
        self._max_shard_bytes = int(self.params.get('max_shard_bytes', 512 << 10))
        self._min_slice_size = int(self.params.get('min_slice_size', 32 << 10))

        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.zeros_initializer()

    def __call__(self, features, labels, mode, params):
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

        auxiliary_info = features['auxiliary_info']
        dense_features = features['dense_feature']
        cat_features = features['cat_feature']
        dense_features = tf.identity(dense_features, name='dense_feature')
        cat_features = tf.identity(cat_features, name='cat_feature')

        base_hashtable = self.initialize_hashtable(key_dtype=tf.int64,
                                                   value_dtype=tf.float32,
                                                   shape=tf.TensorShape(self.embed_dim),
                                                   name='base_hashtable',
                                                   empty_key=tf.int64.max,
                                                   initializer=tf.random_uniform_initializer(minval=-0, maxval=0),
                                                   shard_num=self.ps_num,
                                                   fusion_optimizer_var=True,
                                                   enable_bigmodel=int(self.params['model_export']),
                                                   export_optimizer_var=False)

        # cat_features_embed = self.table_lookup(base_hashtable, is_training, cat_features, 5, params,
        #                                        "embed_base_ids", flatten=False, embedding_size=self.embed_dim)
        ptable_lookup = functools.partial(self.table_lookup, table=base_hashtable, is_training=is_training, threshold=5, params=params, flatten=False, embedding_size=self.embed_dim)
        # self.logger.info('cate_features_embed: %s', cat_features_embed)

        params['logger'] = self.logger
        params['base_hashtable'] = base_hashtable
        params['attention_switch_hub'] = self.attention_switch_hub

        processed_features = []

        def map_one_init(input_unit):
            with tf.name_scope('%s' % input_unit):
                input_class = import_class(input_unit)
                model_class = input_class(self.data_struct, params, is_training, base_hashtable, ptable_lookup)
            return model_class

        def gather_feature(model_classes):
            def one_gather(cat_features, index_of_column, gather_lists):
                # class-> feat_list -> feas
                # cat_list -> [[[1,2],[2,3]],[[2,3,3],[2,3,3]]]
                cat_feat_index = index_of_tensor(index_of_column, list(it.chain(*list(it.chain(*gather_lists)))))
                print("cat_feat_index:", cat_feat_index)
                cat_feat = tf.gather(cat_features, cat_feat_index, axis=1)
                self.logger.info("cat_feat shape is {}".format(cat_feat.get_shape()))

                cat_n_split = list(map(len, list(it.chain(*gather_lists))))
                self.logger.info("cat_n_split is {}".format(cat_n_split))
                feat_split = tf.split(cat_feat, cat_n_split, axis=1)
                self.logger.info("feat_split shape is {}".format(feat_split))
                class_feat_len = list(map(len, list(gather_lists)))
                class_feat_sum = [sum(class_feat_len[:i]) for i in range(len(class_feat_len))]
                feat_split_concat = [feat_split[s: s + l] for s, l in zip(class_feat_sum, class_feat_len)]
                self.logger.info("feat_split_concat is {}".format(feat_split_concat))

                return feat_split_concat

            gather_cat_feas = [m.cat_list for m in model_classes]
            gather_dense_feas = [m.dense_list for m in model_classes]
            cat_split = one_gather(cat_features, model_classes[0].cat_columns_info.index_of_column, gather_cat_feas)
            dense_split = one_gather(dense_features, model_classes[0].dense_columns_info.index_of_column, gather_dense_feas)

            return cat_split, dense_split

        model_classes = map(map_one_init, ut.utils.parse_inputs_unit(params['input_units']))
        cat_split, dense_split = gather_feature(model_classes)
        _ = [m.recieve_gather_features(cat_split[i], dense_split[i]) for i, m in enumerate(model_classes)]

        processed_features = [m(cat_features, dense_features, auxiliary_info, self.embed_dim, self.se_block) for m in model_classes]

        input_layer = tf.concat(processed_features, axis=1)

        self.logger.info('#ZYP processed_features: %s', processed_features)
        self.logger.info('input_layer: %s', input_layer)
        logits = self.inference(input_layer, 'deep_layer', params['activation'], params.get('out_dim', 1), is_training)

        pctr = tf.nn.sigmoid(logits)

        rectified_pctr = pctr / (
                pctr + (1 - pctr) / float(params['sample_rate']))
        tf.identity(rectified_pctr, name='rectified_pctr')

        return logits, pctr, rectified_pctr, []

    def inference(self, input_layer, name_scope, activation, out_dim, is_training):
        with tf.name_scope(name_scope):
            for i, unit_num in enumerate(self.deep_layers):
                net = tf.layers.dense(inputs=input_layer, units=unit_num,
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer,
                                      partitioner=self.get_partitioner('fix'),
                                      name='%s_fc_%d' % (name_scope, i))

                input_layer = tf.layers.batch_normalization(inputs=net, training=is_training,
                                                            name='bn_%d' % i)

                if activation == 'dice':
                    input_layer = self.dice(input_layer, name='dice_%d' % i, is_training=is_training)
                elif activation == 'prelu':
                    input_layer = self.prelu(input_layer, 'fcn_layer_prelu_%d' % i)
                else:
                    input_layer = tf.nn.relu(input_layer)

            net = tf.layers.dense(inputs=input_layer, units=out_dim,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name='%s_output' % name_scope)

            net = tf.layers.batch_normalization(inputs=net, training=is_training,
                                                name='bn_%d' % len(self.deep_layers))
            return net

    def se_block(self, se_input, emb_dim, name_scope, is_training, se_type):
        if se_type == "train_bn":
            se_input = tf.layers.batch_normalization(inputs=se_input, training=is_training,
                                                     name='bn_%s' % name_scope, trainable=True)
        elif se_type == "fix_bn":
            se_input = tf.layers.batch_normalization(inputs=se_input, training=is_training, name='bn_%s' % name_scope,
                                                     trainable=False, beta_initializer=tf.constant_initializer(0.0),
                                                     gamma_initializer=tf.constant_initializer(1.0))
        elif se_type == "no_bn":
            se_input = se_input
        else:
            return se_input

        se_ratio = 4
        fea_num = se_input.shape[1].value / emb_dim
        se_input = tf.reshape(se_input, [-1, fea_num, emb_dim])
        neuron_num = max(1, int(math.ceil(1.0 * fea_num / se_ratio)))

        squeeze_output = tf.reduce_mean(se_input, axis=2)

        with tf.variable_scope("se_block_" + name_scope):
            if neuron_num == 1:
                deep_layer = tf.layers.dense(squeeze_output,
                                             neuron_num,
                                             activation=tf.nn.relu,
                                             name="fc0")
            else:
                deep_layer = tf.layers.dense(squeeze_output,
                                             neuron_num,
                                             activation=tf.nn.relu,
                                             partitioner=self.get_partitioner('var'),
                                             name="fc0")
            if fea_num == 1:
                excitation_out = tf.layers.dense(deep_layer,
                                                 fea_num,
                                                 activation=tf.nn.sigmoid,
                                                 name="fc1")
            else:
                excitation_out = tf.layers.dense(deep_layer,
                                                 fea_num,
                                                 activation=tf.nn.sigmoid,
                                                 partitioner=self.get_partitioner('var'),
                                                 name="fc1")
            excitation_out = tf.expand_dims(excitation_out, axis=-1)
            reweight_out = tf.multiply(excitation_out, se_input)
        output = tf.reshape(reweight_out, [-1, fea_num * emb_dim])
        return output

    def get_partitioner(self, partitioner_type=None):
        partitioner_type = partitioner_type if partitioner_type in self._partitioner_set else self._partitioner
        if partitioner_type == 'fix':
            return tf.fixed_size_partitioner(self.ps_num)
        elif partitioner_type == 'var':
            return tf.variable_axis_size_partitioner(max_shard_bytes=self._max_shard_bytes, max_shards=self.ps_num)
        elif partitioner_type == 'min_max':
            return tf.min_max_variable_partitioner(max_partitions=self.ps_num, min_slice_size=self._min_slice_size)
        else:
            raise RuntimeError('unknown partitioner type %s' % partitioner_type)

    def attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, seq_len_fea_col,
                        embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
                        att_type):
        with tf.name_scope("attention_layer_%s" % (att_type)):
            self.logger.info('cur_poi_seq_fea_col %s', cur_poi_seq_fea_col)
            # self.logger.info("seq_len_fea_col %s", seq_len_fea_col)
            cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])

            # 将query复制 seq_len 次 None, seq_len, embed_dim
            if att_type.startswith('top40'):
                din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
                self.logger.info('#wsl: top20 din is {}'.format(din_all))
            elif att_type == 'click_sess_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            elif att_type == 'order_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            else:
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

            self.logger.info('din_all %s ', din_all)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all

            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             partitioner=self.get_partitioner('min_max'),
                                             name=name_scope + 'f_%d_att' % (i))
                # , reuse=tf.AUTO_REUSE
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
            self.logger.info('din_output_layer %s', din_output_layer)
            din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30
            self.logger.info('din_output_layer %s', din_output_layer)

            # Mask
            if seq_len_fea_col is None:
                outputs = din_output_layer
            else:
                key_masks = tf.sequence_mask(seq_len_fea_col, seq_len)  # [B,1, T] 这个已经是三维的了
                self.logger.info('key_masks %s', key_masks)  # 1, 1, 15

                paddings = tf.zeros_like(din_output_layer)
                ##
                outputs = tf.where(key_masks, din_output_layer, paddings)  # [N, 1, 30]

                tf.summary.histogram("attention", outputs)

            self.logger.info('outputs %s', outputs)

            # 直接加权求和
            weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)

            # [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
            self.logger.info('weighted_outputs %s', weighted_outputs)
            weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])  # N, 8*3
            self.logger.info('weighted_outputs %s', weighted_outputs)
            return weighted_outputs

    def attention_din_nomask(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
                             embed_dim, seq_len, din_deep_layers, din_activation, name_scope,
                             att_type):
        with tf.name_scope("attention_layer_%s" % (att_type)):
            self.logger.info('attention layer')
            cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
            ## AUTOFeas
            scene_feature2 = self.ep_feature_emb2
            scene_output = tf.layers.dense(scene_feature2,
                                           embed_dim,
                                           name=name_scope + "_scene_fc")
            scene_feature2 = tf.tile(scene_output[:, None, :], [1, seq_len, 1])
            # 将query复制 seq_len 次 None, seq_len, embed_dim
            if att_type.startswith('top40'):
                din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
                self.logger.info('#wsl: top20 din is {}'.format(din_all))
            elif att_type == 'click_sess_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, scene_feature2], axis=-1)
            elif att_type == 'order_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, scene_feature2], axis=-1)
            else:
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, scene_feature2], axis=-1)

            self.logger.info('din_all %s ', din_all)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all
            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             partitioner=self.get_partitioner('min_max'),
                                             name=name_scope + 'f_%d_att' % (i))
                # , reuse=tf.AUTO_REUSE
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
            self.logger.info('din_output_layer %s', din_output_layer)
            din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30
            self.logger.info('din_output_layer %s', din_output_layer)
            att_scores = tf.squeeze(din_output_layer, axis=[1])
            # att_scores = tf.squeeze(din_output_layer) ## 非最后一层,不指名axis也没问题!
            # 直接加权求和
            weighted_outputs = tf.matmul(din_output_layer, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)

            # [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
            self.logger.info('weighted_outputs %s', weighted_outputs)
            weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim])  # N, 8*3
            self.logger.info('weighted_outputs %s', weighted_outputs)
            return weighted_outputs, att_scores

    def attention_switch_hub(self, cur_poi_emb, hist_seq_emb, mask_seq, embed_dim, max_seq_len, variable_scope, att_type, switch_on=0):
        if 0 == switch_on:
            return None
        elif 1 == switch_on:
            return self.attention_layer(cur_poi_emb, hist_seq_emb, mask_seq,
                                        embed_dim, 1, max_seq_len, [128, 64], "relu",
                                        variable_scope, att_type)
        elif 2 == switch_on:
            return self.attention_layer(cur_poi_emb, hist_seq_emb, mask_seq,
                                        embed_dim, 3, max_seq_len, [128, 64], "relu",
                                        variable_scope, att_type)


class testDNN(DNN):
    def __init__(self, data_struct, params, logger, **kwargs):
        super(testDNN, self).__init__(data_struct, params, logger, **kwargs)
        self.data_struct = data_struct
        self.params = params
        self.logger = logger

        self.deep_layers = params['deep_layers']
        self.embed_dim = params['embed_dim']
        self.learning_rate = params['learning_rate']
        self.ps_num = params['ps_num']

        self._partitioner_set = {'fix', 'var', 'min_max'}
        _partitioner = self.params.get('partitioner', 'fix')
        self._partitioner = _partitioner if _partitioner in self._partitioner_set else 'fix'
        self._max_shard_bytes = int(self.params.get('max_shard_bytes', 512 << 10))
        self._min_slice_size = int(self.params.get('min_slice_size', 32 << 10))

        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.zeros_initializer()

    def __call__(self, features, labels, mode, params):
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

        auxiliary_info = features['auxiliary_info']
        dense_features = features['dense_feature']
        cat_features = features['cat_feature']
        dense_features = tf.identity(dense_features, name='dense_feature')
        cat_features = tf.identity(cat_features, name='cat_feature')

        base_hashtable = self.initialize_hashtable(key_dtype=tf.int64,
                                                   value_dtype=tf.float32,
                                                   shape=tf.TensorShape(self.embed_dim),
                                                   name='base_hashtable',
                                                   empty_key=tf.int64.max,
                                                   initializer=tf.random_uniform_initializer(minval=-0, maxval=0),
                                                   shard_num=self.ps_num,
                                                   fusion_optimizer_var=True,
                                                   enable_bigmodel=int(self.params['model_export']),
                                                   export_optimizer_var=False)

        # cat_features_embed = self.table_lookup(base_hashtable, is_training, cat_features, 5, params,
        #                                        "embed_base_ids", flatten=False, embedding_size=self.embed_dim)
        ptable_lookup = functools.partial(self.table_lookup, table=base_hashtable, is_training=is_training, threshold=5, params=params, flatten=False, embedding_size=self.embed_dim)
        # self.logger.info('cate_features_embed: %s', cat_features_embed)

        params['logger'] = self.logger
        params['base_hashtable'] = base_hashtable
        params['attention_switch_hub'] = self.attention_switch_hub

        processed_features = []

        def map_one_init(input_unit):
            with tf.name_scope('%s' % input_unit):
                input_class = import_class(input_unit)
                model_class = input_class(self.data_struct, params, is_training, base_hashtable, ptable_lookup)
            return model_class

        def gather_feature(model_classes):
            def one_gather(cat_features, index_of_column, gather_lists):
                # class-> feat_list -> feas
                # cat_list -> [[[1,2],[2,3]],[[2,3,3],[2,3,3]]]
                cat_feat_index = index_of_tensor(index_of_column, list(it.chain(*list(it.chain(*gather_lists)))))
                cat_feat_index = [[x] for x in cat_feat_index]
                self.logger.info("cat_feat_index {}".format(cat_feat_index[:10]))
                self.logger.info("cat_features shape before transpose is {}".format(cat_features.get_shape()))
                cat_features = tf.transpose(cat_features, perm=[1, 0])
                cat_feat = tf.gather_nd(cat_features, cat_feat_index)
                self.logger.info("cat_features shape is {}".format(cat_features.get_shape()))
                self.logger.info("cat_feat shape is {}".format(cat_feat.get_shape()))
                cat_feat = tf.transpose(cat_feat, perm=[1, 0])
                self.logger.info("cat_feat shape after transpose is {}".format(cat_feat.get_shape()))

                cat_n_split = list(map(len, list(it.chain(*gather_lists))))
                self.logger.info("cat_n_split is {}".format(cat_n_split))
                feat_split = tf.split(cat_feat, cat_n_split, axis=1)
                self.logger.info("feat_split shape is {}".format(feat_split))
                class_feat_len = list(map(len, list(gather_lists)))
                class_feat_sum = [sum(class_feat_len[:i]) for i in range(len(class_feat_len))]
                feat_split_concat = [feat_split[s: s + l] for s, l in zip(class_feat_sum, class_feat_len)]
                self.logger.info("feat_split_concat is {}".format(feat_split_concat))

                return feat_split_concat

            gather_cat_feas = [m.cat_list for m in model_classes]
            gather_dense_feas = [m.dense_list for m in model_classes]
            cat_split = one_gather(cat_features, model_classes[0].cat_columns_info.index_of_column, gather_cat_feas)
            dense_split = one_gather(dense_features, model_classes[0].dense_columns_info.index_of_column, gather_dense_feas)

            return cat_split, dense_split

        model_classes = map(map_one_init, ut.utils.parse_inputs_unit(params['input_units']))
        cat_split, dense_split = gather_feature(model_classes)
        _ = [m.recieve_gather_features(cat_split[i], dense_split[i]) for i, m in enumerate(model_classes)]

        processed_features = [m(cat_features, dense_features, auxiliary_info, self.embed_dim, self.se_block) for m in model_classes]

        self.logger.info('#ZYP processed_features: %s', processed_features)
        input_layer = tf.concat(processed_features, axis=1)
        self.logger.info('input_layer: %s', input_layer)
        logits = self.inference(input_layer, 'deep_layer', params['activation'], params.get('out_dim', 1), is_training)

        pctr = tf.nn.sigmoid(logits)

        rectified_pctr = pctr / (
                pctr + (1 - pctr) / float(params['sample_rate']))
        tf.identity(rectified_pctr, name='rectified_pctr')

        return logits, pctr, rectified_pctr, []




