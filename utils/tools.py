# -*- coding:utf-8 -*-

import time,datetime
import math,traceback


def merge(*args):
    # print args,"args"
    m = args[0].copy()
    for i in args:
        m.update(i)

    # print m,"m"
    return m

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

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