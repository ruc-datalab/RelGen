import os
import random
from typing import Dict
from scipy import stats
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import copy
from .column import *


def get_distribution(distinct_values, distribution_type):
	n = len(distinct_values)
	p = 0.5
	mu = 1
	if distribution_type == "binom":
		distribution = stats.binom.pmf(range(n), n, p)
	elif distribution_type == "geom":
		distribution = stats.geom.pmf(range(1, n + 1), p)
	elif distribution_type == "poisson":
		distribution = stats.poisson.pmf(range(n), mu)
	elif distribution_type == "norm":
		distribution = stats.norm.pdf(range(-math.floor(n / 2), math.ceil(n / 2)))
	elif distribution_type == "uniform":
		distribution = [1.0 / len(distinct_values)] * len(distinct_values)
	else:
		raise TypeError("Unknown distribution type")
	assert len(distinct_values) == len(distribution), (distinct_values, distribution)
	return distribution


class Table:
	"""Class for table transformer.

	Class variables:
		name(string): name of table.

		num_column(relgen.table_gen.data.Column): class of numerical column.

		cat_column(relgen.table_gen.data.Column): class of categorical column.

	"""

	def __init__(self, name, metadata: Dict = {}, num_column: Column = DiscreteColumn(), cat_column: Column = OrdinalColumn()):
		self.name = name
		self.num_column = num_column
		self.cat_column = cat_column
		self.column_map = {
			"char": OrdinalColumn(),
			"int": DiscreteColumn(),
			"float": DiscreteColumn(),
			"date": DatetimeColumn(),
			"time": DatetimeColumn(),
			"datetime": DatetimeColumn(),
		}
		self.columns = []
		self.col_dims = []
		self.col_names = []
		self.type_casts = {}
		self.cardinality = None
		self.metadata = metadata
		self.data = None

	def fit(self, dataframe):
		"""Use data to fit table.

		Args:
			dataframe (pandas.Dataframe): Dataframe of the table used to fit the table transformer.

		"""
		dataframe = dataframe.dropna()
		self.data = dataframe.copy()
		# self.table_size = len(dataframe)
		for col in dataframe.columns:
			if "columns" in self.metadata and col in self.metadata["columns"] and "type" in self.metadata["columns"][col]:
				col_type = self.metadata["columns"][col]["type"]
				if col_type == "id" or col_type == "numerical":
					column = copy.deepcopy(self.num_column)
				elif col_type == "categorical":
					column = copy.deepcopy(self.cat_column)
				else:
					raise ValueError("Column type must be id, numerical or categorical")
			else:
				if is_numeric_dtype(dataframe[col]):
					column = copy.deepcopy(self.num_column)
				else:
					column = copy.deepcopy(self.cat_column)
			column.fit_by_instance(col, dataframe[col].values)
			self.columns.append(column)
			self.col_names.append(col)
			self.col_dims.append(column.dim)
		self.cardinality = self.get_cardinality(self.columns)

	def transform(self, dataframe):
		"""
		Transform the given table to the continuous vectors

        Args:
            dataframe(pandas.DataFrame): Dataframe of the table that need to be transformed
		"""

		dataframe = dataframe.dropna()
		trans_data = []
		for col_name in dataframe.columns:
			column = self.columns[self.column_index(col_name)]
			trans_data.append(column.transform(dataframe[col_name].values))
		trans_data = np.concatenate(trans_data, axis=1)
		return trans_data

	def inverse(self, data_instance, columns=None):
		"""
		Inverse the transformed vectors to the original table
		
        Args:
            data_instance(numpy.array): Transformed vectors that need to be inversed
			
			columns(List): Columns of the original table, the vectors will be inversed given column order.
		
		Returns:
			inverse_data(numpy.array): The original tabular data inversed from the vectors.
		"""

		columns = [col.name() for col in self.columns] if columns is None else columns
		inverse_data = []
		sta = 0
		end = 0
		for col_name in columns:
			column = self.columns[self.column_index(col_name)]
			sta = end
			end = end + column.dim
			col_data = data_instance[:, sta:end]
			inverse_data.append(column.inverse(col_data))
		inverse_data = np.concatenate(inverse_data, axis=1)
		inverse_data = pd.DataFrame(inverse_data, columns=columns)
		# inverse_data = inverse_data[self.col_names]
		return inverse_data

	def get_col_data(self, col_id, data_instance):
		''' Returns the column data for a given column index'''

		col_index = np.cumsum(self.col_dims)
		if col_id == 0:
			return data_instance[:, :col_index[0]]
		else:
			return data_instance[:, col_index[col_id-1]:col_index[col_id]]

	@staticmethod
	def get_cardinality(columns):
		"""Checks that all the columns have same the number of rows."""

		cards = [len(c.fit_data) for c in columns]
		c = np.unique(cards)
		assert len(c) == 1, c
		return c[0]

	def column_index(self, name):
		"""Returns index of column with the specified name."""

		name_to_index = {c.name(): i for i, c in enumerate(self.columns)}
		assert name in name_to_index, (name, name_to_index)
		return name_to_index[name]

	def __getitem__(self, column_name):
		# print(self.name_to_index)
		try:
			return self.columns[self.column_index(column_name)]
		except AssertionError:
			return self.columns[self.column_index(self.name + ':' + column_name)]

	def get_primary_key_dataframe(self, primary_key, shuffle=False):
		column = self.columns[self.column_index(primary_key[0])]
		df = pd.DataFrame(column.distinct_values, columns=[column.col_name])
		df["on"] = 1
		for i in range(1, len(primary_key)):
			column = self.columns[self.column_index(primary_key[i])]
			tmp = pd.DataFrame(column.distinct_values, columns=[column.col_name])
			tmp["on"] = 1
			df = pd.merge(df, tmp, how="left", on="on")
		if shuffle:
			df = df.sample(frac=1).reset_index(drop=True)
		return df


def load_tables(names=[], data_dir=".", cols={}, **kwargs):
	"""Load csv, init and fit tables.

	Args:
		names (string): name of csv tables.

		data_dir (string): directory of csv tables.

		cols (dict): columns of tables to load.

		**kwargs: other arguments to load csv.
	"""
	if not data_dir.endswith('/'):
		data_dir += '/'
	tables = {}
	for name in names:
		table = Table(name)
		if name in cols:
			table.fit(pd.read_csv(data_dir + name + ".csv", usecols=cols[name], **kwargs))
		else:
			table.fit(pd.read_csv(data_dir + name + ".csv", **kwargs))
		tables[name] = table
	return tables

