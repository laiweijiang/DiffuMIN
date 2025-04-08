#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
import json
import math
import random
import logging
import importlib
import traceback
import time

import numpy as np
import tensorflow as tf

import CONST as cst
import itertools as it

import datetime


def logloss(ys, ps):
    loss_sum = 0
    for i in range(len(ys)):
        y = ys[i]
        if y > 0:
            y = 1.0
        else:
            y = 0.0

        y1 = ps[i]
        #        y1 = y1*neg_ratio / (1-y1+neg_ratio*y1)
        if y1 >= 1.:
            y1 = 0.999
        elif y1 <= 0.0:
            y1 = 0.001
        y0 = 1 - y1
        try:
            loss_sum += -(y * math.log(y1) + (1 - y) * math.log(y0))
        except Exception as e:
            print('traceback.print_exc():%s' % traceback.print_exc())

            print("log error:" + str(y1) + " " + str(y0) + " " + str(y))
    return loss_sum / len(ys)


def import_class(full_class_name):
    parts = full_class_name.split('.')
    package_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(package_name)
    class_obj = getattr(module, class_name)
    return class_obj


def set_logger():
    logger = logging.getLogger('tensorflow')
    if len(logger.handlers) == 1:
        logger.handlers = []
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        fh = logging.FileHandler('tensorflow.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def init_environment(flags, params, logger):
    if int(params['is_dist']) == 0:
        job_name = 'chief'
        task_index = 0
        ps_num = 1
    else:
        job_name = flags.job_name
        task_index = flags.task_index

        ps_hosts = flags.ps_hosts.split(',')
        ps_num = len(ps_hosts)
        worker_hosts = flags.worker_hosts.split(',')
        chief_hosts = flags.chief_hosts.split(',')
        evaluator_hosts = flags.evaluator_hosts.split(',')
        if len(evaluator_hosts) > 0 and evaluator_hosts[0] != '':
            cluster = {'chief': chief_hosts, 'ps': ps_hosts,
                       'worker': worker_hosts, 'evaluator': evaluator_hosts}
        else:
            cluster = {'chief': chief_hosts, 'ps': ps_hosts,
                       'worker': worker_hosts}

        logger.info('cluster = %s', cluster)

        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster,
             'task': {'type': job_name,
                      'index': task_index}})

    params['ps_num'] = ps_num
    params['job_name'] = job_name
    params['task_index'] = task_index
    params['task'] = flags.task
    params['model_dir'] = flags.model_dir

    logger.info('ps_num : %s' % ps_num)
    logger.info('job_name : %s' % job_name)
    logger.info('task_index : %d' % task_index)


def init_gpu_environment(flags, params, logger):
    if int(params['is_dist']) == 0:
        job_name = 'chief'
        task_index = 0
        ps_num = 0
    else:
        job_name = flags.job_name
        task_index = flags.task_index

        ps_hosts = flags.ps_hosts.split(',')
        ps_num = 0
        worker_hosts = flags.worker_hosts.split(',')
        chief_hosts = flags.chief_hosts.split(',')
        evaluator_hosts = flags.evaluator_hosts.split(',')
        cluster = {"worker": worker_hosts}
        has_evaluators = False

        logger.info('cluster = %s', cluster)

        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster,
             'task': {'type': job_name,
                      'index': task_index}})

    params['ps_num'] = ps_num
    params['job_name'] = job_name
    params['task_index'] = task_index
    params['task'] = flags.task
    params['model_dir'] = flags.model_dir

    logger.info('ps_num : %s' % ps_num)
    logger.info('job_name : %s' % job_name)
    logger.info('task_index : %d' % task_index)


def init_run_config(flags, params, logger):
    if flags.job_name == 'ps':
        device_filters = ['/job:ps', '/job:worker', '/job:master']
    elif flags.job_name == 'chief':
        device_filters = ['/job:ps', '/job:chief']
    elif flags.job_name == 'worker' or flags.job_name == 'evaluator':
        device_filters = ['/job:ps', '/job:%s/task:%d' % (flags.job_name, flags.task_index)]
    else:
        device_filters = []
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    device_filters=device_filters)
    session_config.gpu_options.allow_growth = True

    if int(params['debug']) != 1:
        logger.info('disable_chief_training')
        tf.disable_chief_training(shut_ratio=float(params['shut_ratio']),
                                  slow_worker_delay_ratio=float(params['slow_worker_delay_ratio']))

    tf.enable_persistent_metric()
    has_evaluators = True if len(flags.evaluator_hosts.split(',')) > 0 and flags.evaluator_hosts.split(',')[0] != '' \
        else False
    if 'train' in params['task'] and has_evaluators:
        logger.info('eval_mode=%s', 'train_and_dist_eval')
        run_config = tf.estimator.RunConfig(
            save_summary_steps=int(params['save_summary_steps']),
            save_checkpoints_secs=int(params['save_checkpoints_secs']),
            model_dir=params['model_dir'],
            session_config=session_config,
            keep_checkpoint_max=int(params['keep_checkpoint_max']),
            eval_mode='train_and_dist_eval',
            tf_random_seed=20221218
        )
    else:
        logger.info('eval_mode=%s', 'normal')
        run_config = tf.estimator.RunConfig(
            save_summary_steps=int(params['save_summary_steps']),
            save_checkpoints_secs=int(params['save_checkpoints_secs']),
            model_dir=params['model_dir'],
            session_config=session_config,
            keep_checkpoint_max=int(params['keep_checkpoint_max']),
            tf_random_seed=20221218
        )
    return run_config

def init_gpu_run_config(flags, params, logger):
    if flags.job_name == 'ps':
        device_filters = ['/job:ps', '/job:worker', '/job:master']
    elif flags.job_name == 'chief':
        device_filters = ['/job:ps', '/job:chief']
    elif flags.job_name == 'worker' or flags.job_name == 'evaluator':
        device_filters = ['/job:ps', '/job:%s/task:%d' % (flags.job_name, flags.task_index)]
    else:
        device_filters = []
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    device_filters=device_filters)
    session_config.gpu_options.allow_growth = True

    if int(params['debug']) != 1:
        logger.info('disable_chief_training')
        tf.disable_chief_training(shut_ratio=float(params['shut_ratio']),
                                  slow_worker_delay_ratio=float(params['slow_worker_delay_ratio']))

    tf.enable_persistent_metric()
    has_evaluators = True if len(flags.evaluator_hosts.split(',')) > 0 and flags.evaluator_hosts.split(',')[0] != '' \
        else False
    if 'train' in params['task'] and has_evaluators:
        logger.info('eval_mode=%s', 'train_and_dist_eval')
        run_config = tf.estimator.RunConfig(
            save_summary_steps=int(params['save_summary_steps']),
            save_checkpoints_steps=int(params['save_checkpoints_steps']),
            model_dir=params['model_dir'],
            session_config=session_config,
            keep_checkpoint_max=int(params['keep_checkpoint_max']),
            eval_mode='train_and_dist_eval',
            tf_random_seed=20221218
        )
    else:
        logger.info('eval_mode=%s', 'normal')
        run_config = tf.estimator.RunConfig(
            save_summary_steps=int(params['save_summary_steps']),
            save_checkpoints_steps=int(params['save_checkpoints_steps']),
            model_dir=params['model_dir'],
            session_config=session_config,
            keep_checkpoint_max=int(params['keep_checkpoint_max']),
            tf_random_seed=20221218
        )
    return run_config

def init_single_run_config(flags, params, logger):
    logger.info("job_name=%s", flags.job_name)
    vcore = 5
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True,
                                    inter_op_parallelism_threads=vcore,
                                    intra_op_parallelism_threads=vcore)
    session_config.gpu_options.allow_growth = True

    logger.info("eval_mode=%s", "train_and_dist_eval")
    run_config = tf.estimator.RunConfig(
        save_summary_steps=int(params['save_summary_steps']),
        save_checkpoints_secs=int(params['save_checkpoints_secs']),
        model_dir=params['model_dir'],
        session_config=session_config,
        keep_checkpoint_max=int(params['keep_checkpoint_max']),
    )
    return run_config


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print("*" * 50 + " {} START!!!! ".format(self.process_name) + "*" * 50)
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            duration_seconds = end_time - self.begin_time
            duration = str(datetime.timedelta(seconds=duration_seconds))

            print("#" * 50 + " {} END... time lapsing {}  ". \
                  format(self.process_name, duration) + "#" * 50)
def calc_pred_result(labels, preds):
    labels = np.array(labels)
    labels = labels.astype(np.float64)

    label_mean = labels.mean()
    label_var = labels.var()

    pctr = np.array(preds['rectified_pctr'])
    pctr = pctr.astype(np.float64)
    pctr = np.reshape(pctr, [-1])
    pctr_mean = pctr.mean()
    pctr_var = pctr.var()
    # fpr, tpr, thresholds = metrics.roc_curve(labels, pctr)
    # pctr_auc = metrics.auc(fpr, tpr)
    pctr_auc = approximate_auc_1(labels, pctr)

    loss = logloss(labels, pctr)
    mae = np.abs(labels - pctr).mean()

    result_dict = {}
    result_dict['pctr_auc'] = pctr_auc
    result_dict['loss'] = loss
    result_dict['mae'] = mae
    result_dict['label_mean'] = label_mean
    result_dict['label_var'] = label_var
    result_dict['pctr_mean'] = pctr_mean
    result_dict['pctr_var'] = pctr_var
    return result_dict


def save_eval_result(statistics, flags, params):
    if flags.job_name == 'worker':
        statistics_nodist = statistics.copy()
        statistics_nodist['test_data_start'] = params['test_data_start']
        statistics_nodist['test_data_end'] = params['test_data_end']
        time_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        statistics_file_path = params['model_dir'] + '/{}/testresult'.format(time_prefix)

        if tf.gfile.Exists(statistics_file_path):
            fout = tf.gfile.Open(statistics_file_path, 'a')
        else:
            fout = tf.gfile.Open(statistics_file_path, 'w')

        paras_format = "\n".join(["{}:{}".format(k, v) for k, v in params.items()])
        statistics_format = "\n".join(["{}:{}".format(k, v) for k, v in statistics_nodist.items()])
        fout.write(paras_format + "\n" + statistics_format)
        fout.close()


def get_input_filenames(data_dir):
    file_names = []
    file_list = tf.gfile.ListDirectory(data_dir)
    for current_file_name in file_list:
        file_path = os.path.join(data_dir, current_file_name)
        file_names.append(file_path)
    random.shuffle(file_names)
    return file_names


def approximate_auc_1(labels, preds):
    """
       近似方法，将预测值分桶(n_bins)，对正负样本分别构建直方图，再统计满足条件的正负样本对
       复杂度 O(N)
    """
    # gmv值 换成0、1值
    # labels = np.where(labels > 0.0001, 1.0, 0.0)
    # print("----approximate_auc_1 开始--- %s" % time.asctime(time.localtime(time.time())))
    n_bins = 500000
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 25.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        if nth_bin > n_bins:
            nth_bin = n_bins - 1
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1

    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]
    # print("----approximate_auc_1 结束--- %s" % time.asctime(time.localtime(time.time())))
    return satisfied_pair / float(total_pair + 0.01)


def parse_inputs_unit(input_units,lib_path=cst.handle_lib_path):
    # input "handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense"
    # "input_units": ["handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense"],

    def one_unit(unit):
        f_name, m_names = unit.split(":")
        m_name_list = m_names.split(",")
        return [".".join([lib_path, f_name, m]) for m in m_name_list]

    rs = map(one_unit, input_units)
    return list(it.chain(*rs))

if __name__ == '__main__':
    input_units = ["handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense",
                   "handle_cat_dense_unit2:Dense,Category,PoiTextEmbedDense"]

    print
    parse_inputs_unit(input_units)
