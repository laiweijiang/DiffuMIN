import pandas as pd
# from utils.utils import tick_tock
from utils.tools import tick_tock
from utils.parse_cmd import init_arguments
from utils.tools import approximate_auc_1,logloss
import os
import glob
import os
import json

args = init_arguments()
exp_path = os.path.join(args.exp, "rs")
exp_conf = os.path.join(args.exp, "task_conf.json")


def load_json(config_file):
	params = {}
	data = json.load(open(config_file))
	for k, v in data.iteritems():
		params[k] = v
	return params


def is_non_zero_file(fpath):
	return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def read_one(f):
	print "file_path: {}".format(f)
	flag = is_non_zero_file(f)
	if flag:
		rs = pd.read_csv(f, delimiter='\t')
		rs.columns = ['labels', 'pred']
		# print rs.head(10)
		return rs
	else:
		return None


with tick_tock("read data") as f:
	file_list = glob.glob(exp_path + "/part*")
	dfs = filter(lambda x: x is not None, map(read_one, file_list))
	conf = load_json(exp_conf)
	dfs = pd.concat(dfs, axis=0)
# print dfs.head(10)

with tick_tock("cal auc") as f:
	auc = approximate_auc_1(dfs.labels.tolist(), dfs.pred.tolist())
	logloss = logloss(dfs.labels.tolist(), dfs.pred.tolist())
	print "exp {} final auc is {},logloss is {}".format(args.exp, auc, logloss)
	print "exp {} use conf {}".format(args.exp, " ".join(conf['input_units']))
	print dfs.describe()
