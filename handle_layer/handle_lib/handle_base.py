#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import tensorflow as tf
from data.data_utils import index_of_tensor
import utils.CONST as cst
from tensorflow.keras.layers import *



class GatherFea(object):
	def __init__(self):
		self.dense = []
		self.cat = []

	def append_dense_feas(self, feas_list):
		if isinstance(feas_list, list):
			self.dense = self.dense + feas_list
		else:
			raise Exception("fea_list must be list")

	def append_cat_feas(self, feas_list):
		if isinstance(feas_list, list):
			self.cat = feas_list + self.cat
		else:
			raise Exception("fea_list must be list")


class InputBase(object):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(InputBase, self).__init__()
		self.data_struct = data_struct
		self.params = params
		self.is_training = is_training
		self.logger = params['logger']
		self.hashtable = hashtable
		self.ptable_lookup = ptable_lookup  # ptable_lookup take list_ids,v_name

		self.dense_columns_info = self.data_struct.columns_dict['dense_feature']
		self.cat_columns_info = self.data_struct.columns_dict['cat_feature']
		self.aux_columns_info = self.data_struct.columns_dict['auxiliary_info']
		self.gather_feas = GatherFea()
		self.cat_list = []
		self.dense_list = []

		self.base_hashtable = self.params['base_hashtable']
		self.attention_switch_hub = self.params['attention_switch_hub']

		self.base_embed_dim = 8
		self.r = math.sqrt(6 / self.base_embed_dim)
		self.ALL = "ALL"
		self.NONE = "None"

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

	def eval_str_result(self, inp_list):
		class_paras = self.params.get(cst.class_paras, None)
		paras = class_paras.get(self.__class__.__name__) if class_paras is not None else None
		if paras == self.ALL or paras is None:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return inp_list

		if paras == self.NONE:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return None

		if paras or paras != self.ALL:
			self.out_str = paras.split(",")
			self.logger.info("eval_result work. paras is {}".format(self.out_str))
			return [eval("self." + x) for x in self.out_str]

	def eval_list_result(self, inp_list):
		class_paras = self.params.get(cst.class_paras, None)
		paras = class_paras.get(self.__class__.__name__) if class_paras is not None else None

		if paras == self.ALL or paras is None:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return inp_list

		if paras == self.NONE:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return None

		if paras:
			self.out_str = paras.split(",")
			self.logger.info("eval_result work. paras is {}".format(self.out_str))
			return [inp_list[int(x)] for x in self.out_str]


	def get_cat_emb(self, cat_features, feat_list, feat_name):
		feat_index = index_of_tensor(self.cat_columns_info.index_of_column, feat_list)
		self.logger.info("cat feature shape {}".format(cat_features.get_shape()))
		cat_feat = tf.gather(cat_features, feat_index, axis=1)
		self.logger.info("cat_feat shape {}".format(cat_feat.get_shape()))

		cat_emb = self.ptable_lookup(list_ids=cat_feat, v_name=feat_name)
		return cat_emb


	def get_dense(self, dense_features, feat_list):
		feat_index = index_of_tensor(self.dense_columns_info.index_of_column, feat_list)
		dense_feat = tf.gather(dense_features, feat_index, axis=1)
		return dense_feat

	def easy_attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, mask, att_type):
		cur_name = "att_cur_%s" % att_type
		cur_poi_seq_fea_col = tf.identity(cur_poi_seq_fea_col, name=cur_name)

		hist_name = "att_hist_%s" % att_type
		hist_poi_seq_fea_col = tf.identity(hist_poi_seq_fea_col, name=hist_name)

		if mask is not None:
			seq_len_name = "att_len_%s" % att_type
			mask = tf.identity(mask, name=seq_len_name)
			self.logger.info('gpu tensor name input, attention_layer, input/%s input/%s input/%s' % (cur_name, hist_name, seq_len_name))
		else:
			self.logger.info('gpu tensor name input, attention_layer, input/%s input/%s' % (cur_name, hist_name))

		self.logger.info('easy_attention_layer, cur_poi_seq_fea_col {} hist_poi_seq_fea_col {}'.format(cur_poi_seq_fea_col, hist_poi_seq_fea_col))
		seq_len = hist_poi_seq_fea_col.shape[-2]
		din_deep_layers, din_activation = self.params['din_deep_layers'], self.params['din_activation']
		with tf.name_scope("attention_layer_%s" % att_type):
			cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
			din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

			activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
			input_layer = din_all

			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
				                             name=att_type + 'f_%d_att' % i)
				input_layer = deep_layer

			din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=att_type + 'fout_att')
			din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, seq_len

			# Mask
			if mask is not None:
				if len(mask.shape) == 2:
					if mask.shape[-1] == 1:
						key_masks = tf.sequence_mask(mask, seq_len)  # [B, 1, seq_len]
					else:
						key_masks = tf.expand_dims(mask, -2)
				else:
					key_masks = mask

				outputs = tf.cast(key_masks, tf.float32) * din_output_layer
			else:
				outputs = din_output_layer

			weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, seq_len, (N, seq_len , 24)= (N, 1, 24)

			weighted_outputs = tf.reshape(weighted_outputs, [-1, hist_poi_seq_fea_col.shape[-1]])  # N, 8
			self.logger.info("{} weighted_outputs is {}".format(att_type, weighted_outputs))

			return weighted_outputs

	def attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, seq_len_fea_col,
	                    embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
	                    att_type):
		cur_name = "att_cur_%s" % att_type
		cur_poi_seq_fea_col = tf.identity(cur_poi_seq_fea_col, name=cur_name)

		hist_name = "att_hist_%s" % att_type
		hist_poi_seq_fea_col = tf.identity(hist_poi_seq_fea_col, name=hist_name)

		seq_len_name = "att_len_%s" % att_type
		seq_len_fea_col = tf.identity(seq_len_fea_col, name=seq_len_name)

		self.logger.info('gpu tensor name input, attention_layer, input/%s input/%s input/%s' % (cur_name, hist_name, seq_len_name))

		with tf.name_scope("attention_layer_%s" % att_type):
			cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])

			if att_type.startswith('top40'):
				din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
			elif att_type == 'click_sess_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			elif att_type == 'order_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			else:
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

			activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
			input_layer = din_all

			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
				                             name=name_scope + 'f_%d_att' % i)
				# , reuse=tf.AUTO_REUSE
				input_layer = deep_layer

			din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
			dropout_flag = bool(self.params.get("attention_weights_dropout", 0))
			if dropout_flag:
				din_output_layer = Activation('softmax')(din_output_layer)
				din_output_layer = Dropout(0.2)(din_output_layer)
			din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30

			# Mask
			key_masks = tf.sequence_mask(seq_len_fea_col, seq_len)  # [B,1, T]

			paddings = tf.zeros_like(din_output_layer)
			outputs = tf.where(key_masks, din_output_layer, paddings)  # [N, 1, 30]

			tf.summary.histogram("attention", outputs)

			# 直接加权求和
			weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)

			# [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
			weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])  # N, 8*3

			return weighted_outputs

	def attention_layer_nomask(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
	                           embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
	                           att_type, side_info_seq_fea_col=None):
		cur_name = "att_cur_%s" % att_type
		cur_poi_seq_fea_col = tf.identity(cur_poi_seq_fea_col, name=cur_name)

		hist_name = "att_hist_%s" % att_type
		hist_poi_seq_fea_col = tf.identity(hist_poi_seq_fea_col, name=hist_name)

		self.logger.info('gpu tensor name input, attention_layer, input/%s input/%s' % (cur_name, hist_name))

		with tf.name_scope("attention_layer_%s" % att_type):
			self.logger.info("cur_poi_seq_fea_col {}".format(cur_poi_seq_fea_col.get_shape()))
			cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])

			# 将query复制 seq_len 次 None, seq_len, embed_dim
			if att_type.startswith('top40'):
				din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
				self.logger.info("#ZZZ din_all {}；att_type {} ".format(din_all, att_type))
			elif att_type == 'click_sess_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			elif att_type == 'order_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			else:
				if side_info_seq_fea_col is not None:
					din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, side_info_seq_fea_col], axis=-1)
				else:
					din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			self.logger.info('din_all shape is  {}'.format(din_all.get_shape()))

			activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
			input_layer = din_all

			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
				                             name=name_scope + 'f_%d_att' % i)
				# , reuse=tf.AUTO_REUSE
				input_layer = deep_layer

			din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
			self.logger.info('din_output_layer shape is  {}'.format(din_output_layer.get_shape()))

			dropout_flag = bool(self.params.get("attention_weights_dropout", 0))
			if dropout_flag:
				din_output_layer = Activation('softmax')(din_output_layer)
				din_output_layer = Dropout(0.2)(din_output_layer)

			outputs = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30
			self.logger.info('outputs shape is  {}'.format(outputs.get_shape()))

			# 直接加权求和
			weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)
			# [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
			weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])  # N, 8*3
			return weighted_outputs

	def tf_print(self, tensor, name):
		shape = tensor.shape
		element = 100
		if len(shape) >= 200:
			element = shape[1].value
		tensor = tf.Print(tensor,
		                  [tensor],
		                  first_n=300,
		                  summarize=element,
		                  message="print_" + name)
		return tensor

	def gate_merge_for_list(self, process_features):

		din_deep_layers = [128, len(process_features)]

		# self.target_poi_emb = tf.reshape(self.target_poi_emb, [-1, 1, self.base_embed_dim])

		def create_dense(name, input_layer):
			name_scope = self.__class__.__name__ + "_gate_merge_" + "create_dense"
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
		return return_re

	def create_dense(self, name, input_layer, din_deep_layers):
		name_scope = self.__class__.__name__ + "create_dense"
		for i in range(len(din_deep_layers)):
			deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.sigmoid,
			                             name="_".join([name_scope, name, 'dense_%d' % i]))
			input_layer = deep_layer
		return input_layer

	def gate_merge_for_emb(self, process_features):
		if isinstance(process_features, list):
			concat_rs = tf.concat(process_features, -1)
		else:
			concat_rs = process_features

		din_deep_layers = [64, concat_rs.get_shape().as_list()[-1]]
		self.logger.info("cst_wshape_a {}".format(concat_rs.get_shape()))  # [8000, 5274]
		out_put = self.create_dense('ss', concat_rs, din_deep_layers)  # [b, 5274]
		self.logger.info("cst_wshape_out_put {}".format(out_put.get_shape()))  # [8000, 26]
		rs = tf.multiply(concat_rs, out_put)
		return rs

	def gate_merge_for_dense(self, process_features):

		din_deep_layers = [64, process_features.get_shape().as_list()[-1]]
		self.logger.info("process_features {}".format(process_features.get_shape()))  # [8000, 5274]
		out_put = self.create_dense('ss', process_features, din_deep_layers)  # [b, 5274]
		self.logger.info("out_put {}".format(out_put.get_shape()))  # [8000, 26]
		rs = tf.multiply(process_features, out_put)
		return rs

	def gate_merge_for_emb_cat(self, process_features):
		self.logger.info("process_features {}".format(process_features))  # process_feature [b,feat_num,8]
		feat_num = process_features.get_shape().as_list()[-2]
		din_deep_layers = [256, feat_num]
		concat_rs = tf.reshape(process_features, [-1, self.base_embed_dim * feat_num])  # [b,8*feat_num]

		out_put = self.create_dense('ss', concat_rs, din_deep_layers)  # [b, feat_num]
		self.logger.info("out_put {}".format(out_put))  # process_feature [b,feat_num,8]

		out_put_reshaped = tf.expand_dims(out_put, axis=-1)
		weight_rs = process_features * out_put_reshaped
		self.logger.info("weight_rs {}".format(weight_rs))  # process_feature [b,feat_num,8]
		rs = tf.reshape(weight_rs, [-1, self.base_embed_dim * feat_num])
		return rs

	def gate_merge_for_emb_cat_v2(self, process_features, feat_num, topk):
		self.logger.info("process_features {}".format(process_features))  # process_feature [b,feat_num,8] # [b,35001,8]
		din_deep_layers = [256, feat_num] # [256,35001]
		concat_rs = tf.reshape(process_features, [-1, self.base_embed_dim * feat_num])  # [b,8*feat_num] [500,280008]
		self.logger.info("#LWJ: concat_rs is {}".format(concat_rs))
		out_put = self.create_dense('ss', concat_rs, din_deep_layers)  # [b, feat_num(0.1),initial] (a1,a2,a3,_) [500,35001]
		self.logger.info("#LWJ: out_put is {}".format(out_put))  #  [b,feat_num]  [500, 35001]
		top_k_values, top_k_indices = tf.nn.top_k(out_put, k=topk)

		out_put_reshaped = tf.expand_dims(out_put, axis=-1) #[500,35001,1]
		# top_k_indices = tf.expand_dims(top_k_indices, axis=-1)

		self.logger.info("top_k_indices {}".format(top_k_indices)) # [500,500]
		weight_rs = process_features * out_put_reshaped  #[500, 35001, 8]
		topk_weight_rs = tf.gather(weight_rs[:, :, :], top_k_indices, batch_dims=1) # [500, 500, 8]

		self.logger.info("weight_rs {}".format(weight_rs))
		self.logger.info("topk_weight_rs {}".format(topk_weight_rs))  # process_feature [b,feat_num,8]
		# rs = tf.reshape(weight_rs, [-1, self.base_embed_dim * feat_num])
		return topk_weight_rs

	def gate_merge_for_emb_cat_v3(self, process_features, feat_num, topk):
		din_deep_layers = [256, feat_num]
		concat_rs = tf.reshape(process_features, [-1, 8])  # Reshape to [batch_size*feat_num, 8]
		self.logger.info("process_features {}".format(process_features))  # process_feature [b,feat_num,8]

		out_put = self.create_dense('ss', concat_rs, din_deep_layers)  # [batch_size*feat_num, feat_num(0.1), initial]
		self.logger.info("out_put {}".format(out_put))  # process_feature [b,feat_num,8]

		# Apply Bahdanau Attention
		attention_units = 128
		score = tf.keras.layers.Dense(attention_units, activation='tanh')(out_put)
		score = tf.keras.layers.Dense(1, activation='softmax')(score)
		out_put = tf.reshape(score, [-1, feat_num])  # Reshape to [batch_size, feat_num, feat_num]

		top_k_values, top_k_indices = tf.nn.top_k(out_put, k=topk)

		out_put_reshaped = tf.expand_dims(out_put, axis=-1)
		# top_k_indices = tf.expand_dims(top_k_indices, axis=-1)

		self.logger.info("top_k_indices {}".format(top_k_indices))
		weight_rs = process_features * out_put_reshaped
		topk_weight_rs = tf.gather(weight_rs[:, :, :], top_k_indices, batch_dims=1)

		self.logger.info("weight_rs {}".format(weight_rs))
		self.logger.info("topk_weight_rs {}".format(topk_weight_rs))  # process_feature [b,feat_num,8]
		# rs = tf.reshape(weight_rs, [-1, self.base_embed_dim * feat_num])
		return topk_weight_rs
