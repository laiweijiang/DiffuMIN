#!/usr/bin/python
# -*- coding:utf-8 -*-

# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor


class MagicSearch(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(MagicSearch, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["XXX"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]

		def create_dense(name, input_layer):
			din_deep_layers = [8]
			name_scope = "create_dense"
			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.relu,
				                             name="_".join([name_scope, name, 'dense_%d' % i]))
				input_layer = deep_layer
			return input_layer

		merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]
		self.logger.info("merge_cand {}".format(merge_cand))
		merge_cand = tf.reshape(merge_cand, [-1, feat_len, embed_dim])
		self.logger.info("merge_cand {}".format(merge_cand.get_shape()))
		# merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]

		output = self.attention_layer_nomask(self.target_poi_emb, merge_cand, embed_dim, 1, feat_len, self.params['din_deep_layers'],
		                                     self.params['din_activation'], key, key + '_att')

		return output  # [B, 3*8]


class MagicOutDin(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(MagicOutDin, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["XXX"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]
		din_deep_layers = [64]
		self.target_poi_emb = tf.reshape(self.target_poi_emb, [-1, 1, self.base_embed_dim])

		def create_dense(name, input_layer):

			name_scope = "create_dense"
			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.relu,
				                             name="_".join([name_scope, name, 'dense_%d' % i]))
				input_layer = deep_layer
			return input_layer

		merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]
		rs = []
		for i, cand in enumerate(merge_cand):
			his_cand = merge_cand[:i] + merge_cand[i + 1:]
			cur_cand = cand

			self.logger.info("his_cand len {}".format(len(his_cand)))

			cur_cand = tf.reshape(cur_cand, [-1, 1, din_deep_layers[-1]])
			his_cand = tf.reshape(his_cand, [-1, feat_len - 1, din_deep_layers[-1]])

			self.logger.info("cur_cand {}".format(cur_cand.get_shape()))
			self.logger.info("his_cand {}".format(his_cand.get_shape()))
			output = self.attention_layer_nomask(cur_cand, his_cand, din_deep_layers[-1], 1, feat_len - 1, self.params['din_deep_layers'],
			                                     self.params['din_activation'], "_".join([key, str(i)]), key + '_att')
			rs.append(output)

		return tf.concat(rs, axis=1)  # [B, 3*8]


class Magic2Gate(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(Magic2Gate, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["murmurhash_poi_id_int64"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]
		din_deep_layers = [256, 128, len(process_features)]
		self.target_poi_emb = tf.reshape(self.target_poi_emb, [-1, 1, self.base_embed_dim])

		def create_dense(name, input_layer):
			name_scope = self.__class__.__name__ + key + "create_dense"
			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.sigmoid,
				                             name="_".join([name_scope, name, 'dense_%d' % i]))
				input_layer = deep_layer
			return input_layer

		a = tf.concat(process_features, -1)
		self.logger.info("cst_wshape_a {}".format(a.get_shape()))  # [8000, 5274]
		out_put = create_dense('ss', a)  # [b, 26]
		self.logger.info("cst_wshape_out_put {}".format(out_put.get_shape()))  # [8000, 26]
		out_put = tf.expand_dims(out_put, -1)
		return_re = []  # [b, 26, 1]
		for i, part_feature in enumerate(process_features):
			re_re = tf.multiply(process_features[i], out_put[:, i, :])
			return_re += [re_re]
		return tf.concat(return_re, axis=1)

class Magic2GateDup1(Magic2Gate):
	pass
	# def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
	# 	super(Magic2GateDup1, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
	# 	self.ptable_lookup = self.params.get("{}_ptable_lookup".format("add_table1"))
	# 	# self.logger.info(" {} cat_list {}".format(self.__class__.__name__,self.cat_list))

class Magic2GateDup2(Magic2Gate):
	pass

class Magic2GateByElement(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(Magic2GateByElement, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["XXX"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]
		din_deep_layers = [256, 128, len(process_features)]
		self.target_poi_emb = tf.reshape(self.target_poi_emb, [-1, 1, self.base_embed_dim])

		rs = self.gate_merge_for_emb(process_features)
		return rs
