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
from tensorflow.python.eager import context

from model.model_lib.static import BIG_MODEL_EMBED_HASH_TABLE
from model.model_lib.estimator import Estimator
import numpy as np
from random import randrange

class EstimatorWithFastEvaluate(Estimator):

    def evaluate(self):
        self.logger.info("evaluate_sklearn_dispredict......")
        predictions_iterator = tf.estimator.dist_predict(self.estimator,
                                                         input_fn=self.build_input_fn(),
                                                         yield_single_examples=True)
        line_num = 0
        prediction_values = {'rectified_pctr': [], 'ori_pctr': [], 'labels': []}

        for prediction in predictions_iterator:
            if line_num % 100000 == 0:
                self.logger.info('predict result %d lines, prediction=%s', line_num, prediction)
            line_num += 1

            for key, value in prediction.items():
                if key not in prediction_values:
                    continue
                prediction_values[key].extend(value)
        self.logger.info('dist predict done, line_num=%s', line_num)

        labels = np.expand_dims(prediction_values["labels"], axis=-1)
        rectified_pctr = np.expand_dims(prediction_values["rectified_pctr"], axis=-1)
        # ori_pctr = np.expand_dims(prediction_values["ori_pctr"], axis=-1)
        # self.logger.info("labels.shape {} ,rectified_pctr.shape {} ,ori_pctr.shape {} ".format(labels.shape, rectified_pctr.shape, ori_pctr.shape))
        result = np.concatenate([labels, rectified_pctr], axis=1)
        content = ""
        for label, rec_pctr in result:
            content += "{}\t{}\n".format(label, rec_pctr)
        filename = "{}/rs/part-{}".format(self.params['model_dir'], randrange(10**10))
        self.logger.info("save_pred_result file_name=%s", filename)
        fout = tf.gfile.Open(filename, "w")
        fout.write(content)
        fout.flush()
        fout.close()

        # if self.params['job_name'] == 'worker':
        #     statistics = calc_pred_result(prediction_values['labels'], prediction_values)
        #     self.logger.info('statistics %s', statistics)
        #     save_eval_result(statistics, self.FLAGS, self.params)
        # else:
        #     self.logger.info('not worker evaluate_sklearn_dispredict......')
