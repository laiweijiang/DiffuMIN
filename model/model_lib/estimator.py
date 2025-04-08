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


class Estimator(object):
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

    def _model_fn(self, features, labels, mode, params):
        logits, pctr, rectified_pctr, auxiliary_loss = self.model_class(features, labels, mode, params)

        predictions = {'pctr': pctr,
                       'rectified_pctr': rectified_pctr,
                       'labels': labels}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        labels = tf.reshape(
            tf.cast(labels, tf.float32),
            [-1, 1])
        logits = tf.reshape(logits, [-1, 1])
        loss = self.apply_loss(logits, labels)

        rectified_pctr = tf.reshape(rectified_pctr, [-1, 1])
        eval_cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(rectified_pctr)))

        auc = tf.metrics.auc(labels=labels, predictions=rectified_pctr)
        mae = tf.metrics.mean_absolute_error(labels=labels, predictions=rectified_pctr)
        mean_pctr = tf.metrics.mean(rectified_pctr)
        mean_ctr = tf.metrics.mean(labels)

        _, var_pctr = tf.nn.moments(rectified_pctr, axes=0)
        _, var_ctr = tf.nn.moments(labels, axes=0)

        eval_metric_ops = {'auc': auc,
                           'mae': mae,
                           'avg_loss': eval_cross_entropy,
                           'avg_pctr': mean_pctr,
                           'avg_ctr': mean_ctr,
                           'var_pctr': var_pctr,
                           'var_ctr': var_ctr}

        if mode == tf.estimator.ModeKeys.EVAL:
            tf.summary.scalar('valid-auc', auc[1])
            tf.summary.scalar('valid-loss', loss)
            tf.summary.scalar('valid-mae', mae[1])
            tf.summary.scalar('valid-ctr', mean_ctr[1])
            tf.summary.scalar('valid-pctr', mean_pctr[1])
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        tf.summary.scalar('train-auc', auc[1])
        tf.summary.scalar('train-loss', loss)
        tf.summary.scalar('train-mae', mae[1])
        tf.summary.scalar('train-ctr', mean_ctr[1])
        tf.summary.scalar('train-pctr', mean_pctr[1])

        train_op = self.apply_optimizer(loss, float(params['learning_rate']),
                                        float(params['lazyAdam_learning_rate']), params['optimizer'])

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def init_estimator(self):
        pretrain_model_dir = self.params.get('pretrain_model_dir', '')
        if self.params.get('use_pipeline', False):
            tf.enable_pipeline(data_limit_size=10, data_limit_wait_ms=100)

        if len(pretrain_model_dir) > 10:
            self.logger.info('WarmStartSettings use pretrain model: %s', pretrain_model_dir)
            ws = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=pretrain_model_dir,
                # vars_to_warm_start=['.*'],
                vars_to_warm_start=['base_hashtable'],
                # vars_to_warm_start=['multiworker/dnn.*', 'multiworker/vw']
            )
            return tf.estimator.Estimator(model_fn=self._model_fn, params=self.params,
                                          config=self.run_config, warm_start_from=ws)
        else:
            return tf.estimator.Estimator(model_fn=self._model_fn, params=self.params,
                                          config=self.run_config)

    def build_input_fn(self):
        if int(self.params['is_dist']) == 0 and int(self.params['local_debug']) == 1:
            file_names = get_input_filenames(self.params['train_data_path'])
        elif (int(self.params['is_dist']) == 0 or self.params['job_name'] in ['worker', 'evaluator']) \
             and int(self.params['is_data_dispatch']) == 0:
            file_names = get_input_filenames('inputs')
        else:
            file_names = []

        if int(self.params['is_data_dispatch']) == 1:
            trainset_file_names = None
            self.logger.info('使用AfoDataset')
        else:
            trainset_file_names = file_names
            self.logger.info('trainset_file_names %s', trainset_file_names)

        return functools.partial(self.input_fn, trainset_file_names, self.params['epoch'],
                                 self.params['batch_size'], self.parse_fn,50)

    def train_and_evaluate(self):
        if int(self.params['is_dist']):
            hooks = []
            if int(self.params['debug']) and self.params['job_name'] == 'worker' and self.params['task_index'] == 0:
                profile_hook = tf.train.ProfilerHook(save_steps=500, output_dir=self.params['model_dir'],
                                                     show_memory=False)
                hooks.append(profile_hook)
            train_input_fn = self.build_input_fn()
            train_spec = tf.estimator.TrainSpec(train_input_fn, hooks=hooks)
            eval_spec = tf.estimator.EvalSpec(train_input_fn, throttle_secs=int(4500))
            tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        else:
            self.estimator.train(self.build_input_fn())
            statistics = self.estimator.evaluate(self.build_input_fn())
            save_eval_result(statistics)

    def train(self):
        if int(self.params['is_dist']):
            self.logger.info('training dist..........')
            hooks = []
            if self.params['debug'] and self.params['job_name'] == 'worker' and self.params['task_index'] == 0:
                profile_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=self.params['model_dir'],
                                                     show_memory=False)
                hooks.append(profile_hook)
            train_input_fn = self.build_input_fn()
            train_spec = tf.estimator.TrainSpec(train_input_fn, hooks=hooks)
            eval_spec = tf.estimator.EvalSpec(train_input_fn, throttle_secs=int(4500))
            tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        else:
            self.estimator.train(self.build_input_fn())

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

        if self.params['job_name'] == 'worker':
            statistics = calc_pred_result(prediction_values['labels'], prediction_values)
            self.logger.info('statistics %s', statistics)
            save_eval_result(statistics, self.FLAGS, self.params)
        else:
            self.logger.info('not worker evaluate_sklearn_dispredict......')

    def infer(self):
        predictions_iterator = tf.estimator.dist_predict(self.estimator,
                                                         input_fn=self.build_input_fn(),
                                                         yield_single_examples=True)
        line_num = 0
        prediction_values = {'rectified_pctr': [], 'ori_pctr': [], 'labels': [], 'num_feature': []}

        for prediction in predictions_iterator:
            line_num += 1
            self.logger.info('line: %d, pctr=%s, nf=%s',
                             line_num,
                             prediction['rectified_pctr'],
                             ','.join([str(float(v)) for v in prediction['num_feature']]))
            if line_num > 1000:
                break

            for key, value in prediction.items():
                if key not in prediction_values:
                    continue
                prediction_values[key].append(value)

        check_score_path = self.params['check_score_path']
        fout = tf.gfile.Open(check_score_path, 'w')
        for pctr in prediction_values['rectified_pctr']:
            fout.write(str(pctr[0]) + '\n')
        fout.close()

    def apply_loss(self, y_logits, y_true):
        with tf.name_scope('loss'):
            cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                         logits=y_logits,
                                                                         name='xentropy')
            loss = tf.reduce_mean(cross_entropy_loss, name='xentropy_mean')
        return loss

    def apply_optimizer(self, loss, learning_rate=0.001, lazy_adam_learning_rate=0.0003, optimizer='adam'):
        with tf.name_scope('optimizer'):
            if optimizer == 'adagrad_lazyadam':
                hash_opt = HashAdagradOptimizer(learning_rate)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazy_adam_learning_rate)
            elif optimizer == 'adam_lazyadam':
                hash_opt = HashAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                             use_locking=False, name='HashAdam', shared_beta_power=False,
                                             use_parallel=True)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazy_adam_learning_rate)
            else:
                hash_opt = HashAdagradOptimizer(learning_rate)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazy_adam_learning_rate)
            self.logger.info('hash_opt %s', hash_opt)
            self.logger.info('normal_opt %s', normal_opt)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                var_list_not_in_hashtable = []
                for v in tf.trainable_variables():
                    self.logger.info('[trainable variables] %s %s', v.name, v.shape)
                    if not v.name.startswith('emb_hashtable'):
                        var_list_not_in_hashtable.append(v)
                self.logger.info('HASHTABLE %s', tf.get_collection(BIG_MODEL_EMBED_HASH_TABLE))
                self.logger.info('var_list_not_in_hashtable %s', var_list_not_in_hashtable)

                grads = hash_opt.compute_gradients(loss, var_list=tf.get_collection(BIG_MODEL_EMBED_HASH_TABLE))
                normal_grads = normal_opt.compute_gradients(loss, var_list=var_list_not_in_hashtable)
                hash_train_op = hash_opt.apply_gradients(grads)
                normal_train_op = normal_opt.apply_gradients(normal_grads, global_step=tf.train.get_global_step())

                train_op = tf.group(hash_train_op, normal_train_op)

                return train_op



