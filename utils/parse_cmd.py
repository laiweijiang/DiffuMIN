import pandas as pd
import numpy as np
# import lightgbm as lgbm

from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tools import tick_tock
import argparse
import glob
import itertools


def init_arguments():
    def str_to_bool(s):
        if s == 'true':
            return True
        elif s == 'false':
            return False
        else:
            raise ValueError

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, dest="exp", default="./exp/user/base",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--utype', type=str, dest="utype", default="./exp/user/base",
                        help="map key(rider or rider_poi)")

    parser.add_argument('--file', type=str, dest="file", default="./exp/user/base",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--intro', type=str, dest="intro", default="base",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--valid', type=str_to_bool, dest="valid", default=False,
                        help="map key(rider or rider_poi)")

    parser.add_argument('--fea', type=str, dest="feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--cat', type=str, dest="cat", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--rm', type=str, dest="rm_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--paras', type=str, dest="paras",
                        default="num_boost_round,2000==objective,regression_l2==metric,None=boosting_type,gbdt==learning_rate,0.15==num_leaves,100==max_depth,15==feature_fraction,0.7==min_child_samples,100==nthread,12==filter_thred,50000",
                        help="map key(rider or rider_poi)")

    parser.add_argument('--addfeas', type=str, dest="add_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--labeltran', type=str, dest="label_tran", default="tran_method,expk==value,1.0",
                        help="map key(rider or rider_poi)")

    parser.add_argument('--detailfeas', type=str, dest="detail_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--expk', type=int, dest="expk", default=4,
                        help=" expk for target transform")
    parser.add_argument('--shift', type=int, dest="shift", default=-1,
                        help=" shift for target transform")

    return parser.parse_args()


def safe_float(x):
    try:
        x = float(x)
    except:
        return x
    try:
        if float(x) != int(x):
            return float(x)
        elif float(x) == int(x):
            return int(x)
    except:
        return x


def parse_paras(args):
    paras = args.paras if len(args.paras) > 1 else None
    print("input paras", paras)
    p_dict = {x.split(",")[0]: safe_float(x.split(",")[1]) for x in paras.split("==")}
    print("p_dict", p_dict)
    return p_dict


def parse_label_tran_cmd(args):
    label_tran = args.label_tran if len(args.label_tran) > 1 else None
    p_dict = {x.split(",")[0]: safe_float(x.split(",")[1]) for x in label_tran.split("==")}
    print("label tran,", p_dict)
    return p_dict


def add_feature(args, train_df, test_df, nrows=None):
    rm_feas = args.rm_feas.split(",") if len(args.rm_feas) > 1 else []
    feas = args.feas.split(",") if len(args.feas) > 1 else []
    cat = args.cat.split(",") if len(args.cat) > 1 else []
    add_feas_files = args.add_feas.split(",") if len(args.add_feas) > 1 else None
    detail_feas_list = args.detail_feas.split("=====") if len(args.detail_feas) > 1 else None

    # print("predictors : ", predictors, len(predictors))
    # predictors = list(set(args.feas.split(",")) - set(rm_feas))
    # print('after rm ', predictors, len(predictors))
    if add_feas_files:
        with tick_tock("start add feature"):
            for n, af in enumerate(add_feas_files):
                if len(af) > 1:
                    print("add feature group", af)
                    # if len(af) > 3:
                    for_cols = pd.read_csv("../data/train_" + af + ".csv", nrows=5)
                    to_add_cols = list(for_cols.columns)
                    if detail_feas_list:
                        be_added = detail_feas_list[n].split(",")

                        if "*" in be_added[0]:
                            be_added = to_add_cols

                        cmd = "_".join(be_added[0].split("_")[1:])

                        if cmd.startswith("add"):
                            pt = be_added[0]
                            pts = pt.split("_")[1:]

                            select_cols = []
                            for p in pts:
                                sel = filter(lambda x: p in x, to_add_cols)
                                print("patten", p, " add ", sel)
                                select_cols.append(sel)
                            f_sel_cols = list(set(itertools.chain(*select_cols)))
                            be_added = f_sel_cols

                    else:
                        be_added = to_add_cols

                    print('add fea:', be_added)
                    train_to_append = pd.read_csv("../data/train_" + af + ".csv", usecols=be_added, nrows=nrows)
                    test_to_append = pd.read_csv("../data/test_" + af + ".csv", usecols=be_added, nrows=nrows)

                    print(be_added, "{} be_added".format(len(be_added)))
                    feas += be_added
                    # be_added
                    # = list(train_to_append.columns)
                    # be_added = list(set(be_added) - set([u'minute', u'second']))
                    # gc.collect()
                    print(len(train_df), len(train_to_append))
                    print(len(test_df), len(test_to_append))

                    assert len(train_df) == len(train_to_append)
                    assert len(test_df) == len(test_to_append)

                    train_df = train_df.join(train_to_append[be_added])
                    test_df = test_df.join(test_to_append[be_added])
                    del train_to_append
                    del test_to_append
                    import gc
                    gc.collect()

    con = list(set(feas) - set(rm_feas) - set(cat))

    print("len is ", len(con), "final con is ", con)
    print("len is ", len(cat), "final cat is ", cat)

    return train_df, test_df, con, cat


def add_models(args, stacking_data_root="../tgt_stacking/", nrows=None):
    feas = args.add_feas.split(",") if len(args.feas) > 1 else None
    for_stacking = []
    for file in glob.glob(stacking_data_root + "*.csv"):
        for_stacking.append(file)
    train_list = []
    test_list = []
    print("for_stacking", for_stacking)

    train = pd.read_csv("../data/train_base.csv")
    test = pd.read_csv("../data/test_base.csv")

    train = train[train.sales < 50000]

    print('train[train.sales < 50000]', len(train))
    train_len = len(train)
    test_len = len(test)

    train_list.append(train)
    test_list.append(test)

    if feas:
        with tick_tock("start add model"):
            for n, af in enumerate(feas):
                if len(af) > 1:
                    print(af, "af")
                    # if len(af) > 3:

                sort = sorted(for_stacking)
                train_file = filter(lambda x: "cv" in x and af in x, sort)[0]
                test_file = filter(lambda x: "cv" not in x and af in x, sort)[0]

                print("add train file {} , test file {}".format(train_file, test_file))

                a = pd.read_csv(stacking_data_root + train_file, usecols=['sales'])
                b = pd.read_csv(stacking_data_root + test_file, usecols=['sales'])

                a = a.rename(columns={"sales": af})
                b = b.rename(columns={"sales": af})

                # print "train.head()",a.head()
                print(af)
                print("train mean {}, test mean {} ,train std {}, test std {}".format(a[af].mean(), b[af].mean(),
                                                                                      a[af].std(), b[af].std()))
                print("len(train) {} ,len(test) {} ".format(len(a), len(b)))
                assert len(a) == train_len
                assert len(b) == test_len

                train_list.append(a)
                test_list.append(b)

    print(map(lambda x: len(x), train_list))
    print(list(train.columns) + map(lambda x: list(x.columns)[0], train_list[1:]))
    train_df = pd.DataFrame(np.hstack(train_list),
                            columns=list(train.columns) + map(lambda x: list(x.columns)[0], train_list[1:]))
    test_df = pd.DataFrame(np.hstack(test_list),
                           columns=list(test.columns) + map(lambda x: list(x.columns)[0], test_list[1:]))

    # train_df = pd.concat(train_list, axis=1,ignore_index=True)
    print(train_df.head())
    print(test_df.head())

    # test_df = pd.concat(test_list, axis=1,ignore_index=True)

    print(len(train_df), "len(train_df)")
    print(train_len, "train_len")
    assert len(train_df) == train_len
    assert len(test_df) == test_len

    print(train_df.columns)

    return train_df, test_df, feas


if __name__ == '__main__':
    print(type(safe_float("0.5")))
    print(type(safe_float("1.0")))
    print(type(safe_float("1.1")))

    # print float("0.5"),type(floa)
