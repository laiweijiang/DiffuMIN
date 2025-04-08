#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
from data.data_struct import DataStruct


class BaseConfig(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj)


class TaskConfig(object):
    def __init__(self, config_file, data_struct_file):
        self.config_file = config_file
        self.params = self.init_conf()
        self.data_struct = DataStruct(data_struct_file)

    def init_conf(self):
        params = {}
        data = json.load(open(self.config_file))
        for k, v in data.iteritems():
            params[k] = v
        return params

