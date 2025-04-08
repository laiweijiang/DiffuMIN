#!/usr/bin/python
# -*- coding:utf-8 -*-

import functools

import tensorflow as tf
from tensorflow.contrib.opt import HashAdamOptimizer, HashAdagradOptimizer
from data.data_utils import input_fn
from data.data_utils import parse_fn

from utils.utils import calc_pred_result
from utils.utils import save_eval_result
from utils.utils import get_input_filenames

from model.model_lib.static import BIG_MODEL_EMBED_HASH_TABLE
from model.model_lib.estimator import Estimator
from tensorflow.python.eager import context

import numpy as np


class GPUEstimator(Estimator):
    def __init__(self, model_class, data_struct, run_config, logger, FLAGS, params):
        self.model_class = model_class
        self.data_struct = data_struct
        self.run_config = run_config
        self.params = params
        self.FLAGS = FLAGS
        self.logger = logger
        self.input_fn = input_fn
        self.parse_fn = lambda example: parse_fn(example, self.data_struct)
        self.estimator = self.init_estimator()

    def build_input_fn(self):
        if (int(self.params['is_dist']) == 0 or self.params['job_name'] in ['worker', 'evaluator']) \
                and int(self.params['is_data_dispatch']) == 0:
            file_names = get_input_filenames('inputs')
        else:
            file_names = []

        if int(self.params['is_data_dispatch']) == 1:
            trainset_file_names = None
            self.logger.info('AfoDataset')
        else:
            trainset_file_names = file_names
            self.logger.info('trainset_file_names %s', trainset_file_names)

        batch_size = self.params['batch_size'] if self.FLAGS.task == 'train' else self.params['eva_batch_size']
        return functools.partial(self.input_fn, trainset_file_names, self.params['epoch'],
                                 batch_size, self.parse_fn, 50)

    def evaluate(self):
        self.logger.info("gpu_predict......")
        predictions_iterator = tf.estimator.gpu_predict(self.estimator,
                                                        input_fn=self.build_input_fn(),
                                                        yield_single_examples=True)
        line_num = 0
        prediction_values = {"rectified_pctr": [], "ori_pctr": [], "labels": []}
        for prediction in predictions_iterator:
            if line_num % 100000 == 0:
                self.logger.info("predict result %d lines, prediction=%s", line_num, prediction)
                pass
            line_num += 1
            for key, value in prediction.items():
                if key not in prediction_values:
                    continue
                prediction_values[key].extend(value)
        self.logger.info("gpu_predict done, line_num=%s", line_num)
        labels = np.expand_dims(prediction_values["labels"], axis=-1)
        rectified_pctr = np.expand_dims(prediction_values["rectified_pctr"], axis=-1)
        # ori_pctr = np.expand_dims(prediction_values["ori_pctr"], axis=-1)
        # self.logger.info("labels.shape {} ,rectified_pctr.shape {} ,ori_pctr.shape {} ".format(labels.shape, rectified_pctr.shape, ori_pctr.shape))
        result = np.concatenate([labels, rectified_pctr], axis=1)
        content = ""
        for label, rec_pctr in result:
            content += "{}\t{}\n".format(label, rec_pctr)
        filename = "{}/rs/part-{:05d}".format(self.params['model_dir'], context.context().task_id)
        self.logger.info("save_pred_result file_name=%s", filename)
        fout = tf.gfile.Open(filename, "w")
        fout.write(content)
        fout.flush()
        fout.close()
        succ_path = "{}/pred_succ".format(self.params['model_dir'])
        self.touch_success(succ_path)
        self.wait_success(succ_path)

    # self.logger.info("evaluate_sklearn done, line_num=%s", line_num)

    def touch_success(self, path):
        fout_result = tf.io.gfile.GFile(
            path + "/" + "%s.success" % (self.FLAGS.task_index), "w")
        fout_result.write('success\n')
        fout_result.close()

    def wait_success(self, path):
        self.wait_path(path)

    def wait_path(self, path):
        files = []
        while True:
            files = tf.io.gfile.listdir(path)
            if len(files) == 8:
                break
