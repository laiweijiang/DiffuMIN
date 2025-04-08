#!/usr/bin/python
# -*- coding:utf-8 -*-

import json


class ColumnInfo:
    def __init__(self):
        self.name = ''
        self.columns = []
        self.index_of_column = {}
        self.type = int

    @staticmethod
    def create(info):
        if not isinstance(info, dict):
            return None

        column = ColumnInfo()
        column.columns = info['column']
        column.index_of_column = {}
        for index, col in enumerate(column.columns):
            column.index_of_column[col] = index
        type_str = info['type']
        column.type = type_str
        return column

    def set_columns(self, columns):
        self.columns = columns
        for index, col in enumerate(columns):
            self.index_of_column[col] = index

    def get_index(self, col):
        return self.index_of_column.get(col, -1)


class DataStruct(object):
    def __init__(self, filename):
        self.filename = filename
        self.columns_dict = {}
        self.load()

    def load(self):
        data = json.load(open(self.filename))
        for key, values in data.iteritems():
            if key != 'features':
                continue
            for k, v in values.iteritems():
                column = ColumnInfo.create(v)
                if column is None:
                    continue
                column.name = k
                self.columns_dict[k] = column
        return self


class EmbeddingConfig:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.type = ''
        self.mean = 0.0
        self.stddev = 0.0001


class EmbedFeatList:
    def __init__(self, name, feat_list):
        self.embed_name = name
        self.feat_list = feat_list


class EmbedFeat:
    def __init__(self, name, feat, config={}, scope=''):
        self.embed_name = name
        self.feat = feat
        self.config = config
        self.scope = scope
