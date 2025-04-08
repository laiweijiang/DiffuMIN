#!/usr/bin/python
# -*- coding:utf-8 -*-

import traceback

from utils.utils import *
from utils.config import TaskConfig
from model.model_lib.GpuEstimator import GPUEstimator

import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS


def main(_):
    logger = set_logger()

    os.environ['TRAINER_UNIQUE_THREAD_NUM'] = '5'

    os.environ['TF_XLA_NORMAL_PERF_STEPS'] = '500'
    os.environ['TF_XLA_POLICY_PERF_STEPS'] = '500'
    os.environ['TF_XLA_DEBUG_MODE_PERF'] = '1'
    pass_size = 50 if FLAGS.task == 'train' else 1

    tf.enable_gpu_booster(mode=3, enable_embed_pipeline=False, enable_xla=True, enable_hashtable_fusion=True,
                          sparse_feature_encode_shift=60, sparse_feature_encode_with_mod=True,
                          trainer_pass_size=pass_size, pass_buffer_size=2, batch_prefetch_buffer_size=50
                          )
    tf.xla_warmup_options(
        xla_warmup_steps=1000,
        xla_max_variations=1,
        xla_min_computations=500,
    )

    task_config = TaskConfig(FLAGS.config_file, FLAGS.data_struct_file)
    init_gpu_environment(FLAGS, task_config.params, logger)

    model_class = import_class(task_config.params['model_class_name'])
    model = model_class(task_config.data_struct,
                        task_config.params,
                        logger,
                        **task_config.params)

    run_config = init_gpu_run_config(FLAGS, task_config.params, logger)
    estimator = GPUEstimator(model, task_config.data_struct, run_config,
                             logger, FLAGS, task_config.params)

    try:
        if FLAGS.task == 'train':
            estimator.train()
        elif FLAGS.task == 'evaluate':
            estimator.evaluate()
        elif FLAGS.task == 'infer':
            estimator.infer()
        else:
            raise ValueError('Run task not exist!')
    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:%s\n%s' % (e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            logger.warn(msg)
        else:
            logger.error(msg)
        # 0 means 'be over', 1 means 'will retry'
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)


if __name__ == "__main__":
    tf.app.run()
