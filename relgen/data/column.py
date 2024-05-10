import sys
import re
import string
import random

import pandas as pd
import numpy as np 
from .data_transformer import *


def exists(x):
	return False if x is None else True


class Column:
	""" Abstract class for column transformers
		
		Define some basic operations used in column transformation.
	
		Class variables:
			col_name (str): Name of the column

			dim (int): Dimension of the transformed column

			col_type (str): Type of the column

			is_fit (bool): Whether the column is fitted or not
	"""

	def __init__(self):
		self.col_name = None
		self.dim = None
		self.col_type = None
		self.is_fit = False

	def dim(self):
		""" Get the dimension of the transformed column 

			Returns:
				dim(int): dimenstion of the column
		"""

		if exists(self.dim):
			return self.dim
		else:
			raise ValueError

	def name(self):
		""" Get the column name """
		return self.col_name

	def fit_by_instance(self, col_name, data_instance):
		""" Fit this column instance to the column transformer """

		raise NotImplementedError

	def transform(self, data):
		""" Transform the give column data """
		raise NotImplementedError

	def inverse(self, data):
		""" Inverse the given transformed data to the original data """
		raise NotImplementedError

	def get_null_index(self, data_instance):
		""" Get the index of row containing null value 
			Args:
				data_instance(pandas.DataFrame): Columns instance
			Returns:
				null_index(numpy.array): Index of row containing null value 
		"""

		null_index = np.where(pd.isna(data_instance))[0]
		return null_index 

	def get_exist_index(self, data_instance):
		""" Get the index of row not null 

			Args:
				data_instance(pandas.DataFrame): Columns instance
			Returns:
				null_index(numpy.array): Index of row not null
		"""

		exist_index = np.where(pd.notna(data_instance))[0]
		return exist_index 

	def fit_transform(self, col_name: str, data_instance: pd.DataFrame):
		""" Fit the column transformer and then transform the given columns instance

			Args:
				col_name(str): Column name
				data_instance(pandas.DataFrame): Column instance
			Returns:
				transformed_column(numpy.array): Transformed column
		"""

		self.fit_by_instance(col_name, data_instance)
		return self.transform(data_instance)


class NumericalColumn(Column):
	"""
		Abstract class for numerical column transformers	

		Class variables:
			data_transformer (ColumnTransformer): Transformer used to transform the column

			min (float): The minimum value of the column

			max (float): The maximum value of the column

			dtype (numpy.dtype): Data type of the column
	"""

	def __init__(self):
		super().__init__()
		self.data_transformer = None 
		self.min = None 
		self.max = None 
		self.dtype = None

	def fit_by_instance(self, col_name, data_instance):
		"""
        Args:
            col_name(string): Column name.

            data_instance(pandas.DataFrame): Column instance
		"""

		self.col_name = col_name
		self.fit_data = data_instance
		data_instance = data_instance[pd.notna(data_instance)]
		data_instance = data_instance.reshape(-1, 1)
		self.dtype = data_instance.dtype
		self.min = data_instance.min()
		self.max = data_instance.max()
		self.data_transformer.fit(data_instance)
		self.is_fit = True
	
	def transform(self, data_instance):
		"""
        Args:
            data_instance(pandas.DataFrame): Column instance that need to be transformed 
		"""


		data_instance = data_instance.reshape(-1, 1)
		return self.data_transformer.transform(data_instance)

	def inverse(self, data_instance):
		""" Inverse the transformed column to the original data instance """

		assert self.is_fit
		inverse_data = self.data_transformer.inverse_transform(data_instance)
		if self.dtype == np.int32 or self.dtype == np.int64:
			inverse_data = np.round(inverse_data)
		inverse_data = inverse_data.astype(self.dtype)
		return inverse_data

	def __repr__(self):
		if self.is_fit:
			return f"{self.col_type}Column({self.col_name}, Min={self.min}, Max={self.max}, dtype={self.dtype}, Dim={self.dim})"
		else:
			return f"{self.col_type}Column({self.col_name}), Not fit yet."

class QuantileColumn(NumericalColumn):
	"""Class for numerical column transformers implemented with quantile transformer"""

	def __init__(self, seed=0, subsample=1e9, output_distribution='normal', bins=30):
		super().__init__()
		self.data_transformer = RelQuantileTransformer(
			seed = seed, 
			subsample = subsample,
			output_distribution = output_distribution,
			bins = bins
		)
		self.dim = 1
		self.col_type = "Quantile" 

class ScaleColumn(NumericalColumn):
	"""Class for numerical column transformers implemented with scale transformer"""

	def __init__(self, scaler="MinMax", feature_range=(0, 1)):
		super().__init__()
		self.col_type = scaler
		self.data_transformer = ScaleTransformer(scaler, feature_range) 
		self.dim = 1

class GaussianMixtureColumn(NumericalColumn):
	"""Class for numerical column transformers implemented with Gaussian mixture transformer"""

	def __init__(self, n_component=5):
		super().__init__()
		self.data_transformer = GaussianMixtureTransformer(n_component)
		self.n_component = n_component
		self.dim = n_component + 1
		self.col_type = "GaussianMixture"


class CategoricalColumn(Column):

	"""
		Abstract class for numerical column transformers	

		Attributes:
			distinct_values (numpy.array): Distinct values of the columns

			distribution_size (int): Domain size of the column
	"""

	def __init__(self):
		super().__init__()
		self.distinct_values = None 
		self.distribution_size = None 
	
	def get_distinct_values(self, data_instance):
		"""
		Get the distinct values of the given column instance

        Args:
            data_instance(pandas.DataFrame): Column instance
		"""

		data_instance = data_instance[pd.notna(data_instance)]
		self.distinct_values = np.unique(data_instance)
		self.distinct_values.sort()
		self.distribution_size = len(self.distinct_values)

	def cat_to_index(self, data_instance):
		"""
		Transform the values in the given column instance to corresponding ordinal index

        Args:
            data_instance(pandas.DataFrame): Column instance
        Returns:
            index_data(numpy.array): Transformed result
		"""


		assert exists(self.distinct_values) and exists(self.distribution_size)
		index_data = pd.Categorical(data_instance, categories=self.distinct_values).codes
		return index_data

	def index_to_cat(self, data_instance):
		"""
		Inverse the transformed ordinal index to the original categorical values

        Args:
            data_instance(numpy.array): transformed data instance
        Returns:
            cat_data(numpy.array): inversed result
		"""

		assert exists(self.distinct_values) and exists(self.distribution_size)
		values = np.clip(data_instance, 0, len(self.distinct_values)-1)
		cat_data = np.zeros_like(data_instance).astype(object)
		for i, val in enumerate(values):
			cat_data[i] = self.distinct_values[int(val)]
		return cat_data

	def insert_null_in_domain(self):
		# Convention: np.nan would only appear first.
		if not pd.isnull(self.distinct_values[0]):
			if self.distinct_values.dtype == np.dtype('object'):
				# String columns: inserting nan preserves the dtype.
				self.distinct_values = np.insert(self.distinct_values, 0, np.nan)
			else:
				# Assumed to be numeric columns.  np.nan is treated as a
				# float.
				self.distinct_values = np.insert(
					self.distinct_values.astype(np.float64, copy=False), 0,
					np.nan)
			self.distribution_size = len(self.distinct_values)

	def __repr__(self):
		if self.is_fit:
			return f"{self.col_type}Column({self.col_name}, Domain_size={len(self.distinct_values)}, Dim={self.dim})"
		else:
			return f"{self.col_type}Column({self.col_name}), Not fit yet."


class DiscreteColumn(CategoricalColumn):
	"""Class for discrete column transformers	
	"""

	def __init__(self):
		super().__init__()
		self.col_type = "Discrete"
		self.dtype = None # float or int
	
	def fit_by_instance(self, col_name, data_instance):
		"""
        Args:
            col_name(string): column name.

            data_instance(pandas.DataFrame): dataframe of the column instance used to fit the transformer
		"""

		self.col_name = col_name
		self.fit_data = data_instance
		self.get_distinct_values(data_instance)
		self.dim = 1
		self.is_fit = True
	
	def fit_by_metadata(self, metadata):
		"""
        Args:
            metadata(dict): metadata of the column
		"""

		self.col_name = metadata["name"]
		self.min = float(metadata["min"])
		self.max = float(metadata["max"])
		self.distribution_size = int(metadata["value_size"])
		self.distinct_values = np.unique(metadata["values"]).astype(np.float64)
		if metadata["type"] == 'int':
			self.distinct_values = self.distinct_values.astype(np.int64)
		if metadata["type"] == 'int':
			self.dtype = np.int64
			step = max(int((self.max - self.min) / self.distribution_size), 1)
		else:
			self.dtype = np.float64
			step = (self.max - self.min) / self.distribution_size
		item = self.min
		while len(self.distinct_values) < self.distribution_size:
			if item not in self.distinct_values:
				self.distinct_values = np.append(self.distinct_values, item)
			item += step
		self.distinct_values.sort()
		if "not_null" in metadata and metadata["not_null"] is False:
			if metadata["type"] == 'int':
				self.distinct_values = np.append(self.distinct_values, np.iinfo(np.int32).max)
			else:
				self.distinct_values = np.append(self.distinct_values, float("inf"))
			self.distribution_size += 1
		if metadata["type"] == 'int':
			self.distinct_values = self.distinct_values.astype(np.int64)
		else:
			self.distinct_values = self.distinct_values.astype(np.float64)
		assert len(self.distinct_values) == self.distribution_size
		self.dim = 1
		self.is_fit = True
	
	def transform(self, data_instance):
		index_data = self.cat_to_index(data_instance)
		index_data = index_data.reshape(-1, self.dim)
		if -1 in index_data:
			data_instance = [find_closest_value(element, self.distinct_values) for element in data_instance]
			index_data = self.cat_to_index(data_instance)
			index_data = index_data.reshape(-1, self.dim)
		return index_data 

	def inverse(self, data_instance):
		cat_data = self.index_to_cat(data_instance)
		cat_data = cat_data.reshape(-1, 1)
		return cat_data

	def set_factorization_fields(self, factor_id=None, bit_width=None, bit_offset=None, domain_bits=None, num_bits=None):
		# Factorization related fields.
		self.factor_id = factor_id
		self.bit_width = bit_width
		self.bit_offset = bit_offset
		self.domain_bits = domain_bits
		self.num_bits = num_bits


class OrdinalColumn(CategoricalColumn):
	"""Class for ordinal column transformers	
	"""

	def __init__(self):
		super().__init__()
		self.col_type ="Ordinal"

	def fit_by_instance(self, col_name, data_instance):
		"""
        Args:
            col_name(string): column name.
            
            data_instance(pandas.DataFrame): dataframe of the column instance used to fit the transformer
		"""

		self.col_name = col_name
		self.fit_data = data_instance
		self.get_distinct_values(data_instance)
		self.dim = 1
		self.is_fit = True 
	
	def fit_by_metadata(self, metadata):
		"""
        Args:
            metadata(dict): metadata of the column
		"""

		self.col_name = metadata["name"]
		self.dtype = np.object
		self.distribution_size = int(metadata["value_size"])
		self.distinct_values = np.unique(metadata["values"]).astype(np.object)
		self.distinct_values.sort()
		# i = 0
		# while len(self.distinct_values) < self.distribution_size:
		# 	self.distinct_values = np.append(self.distinct_values, self.col_name + '_' + str(i))
		# 	i += 1
		pattern = re.compile(r"char\((\d+)\)", re.I)
		match = pattern.match(metadata["type"])
		if match:
			length = int(match.groups()[0])
		else:
			length = len(self.col_name)
		assert self.distribution_size - len(self.distinct_values) <= pow(len(string.ascii_letters + string.digits),
																		 length)
		while len(self.distinct_values) < self.distribution_size:
			str_list = [random.choice(string.ascii_letters + string.digits) for i in range(length)]
			random_str = ''.join(str_list)
			if random_str not in self.distinct_values:
				self.distinct_values = np.append(self.distinct_values, random_str)
		if "not_null" in metadata and metadata["not_null"] is False:
			self.distinct_values = np.append(self.distinct_values, "NULL")
			self.distribution_size += 1
		self.distinct_values = self.distinct_values.astype(np.object)
		assert len(self.distinct_values) == self.distribution_size
		self.dim = 1
		self.is_fit = True

	def transform(self, data_instance):
		index_data = self.cat_to_index(data_instance)
		index_data = index_data.reshape(-1, self.dim)
		return index_data 

	def inverse(self, data_instance):
		cat_data = self.index_to_cat(data_instance)
		cat_data = cat_data.reshape(-1, 1)
		return cat_data

	def set_factorization_fields(self, factor_id=None, bit_width=None, bit_offset=None, domain_bits=None, num_bits=None):
		# Factorization related fields.
		self.factor_id = factor_id
		self.bit_width = bit_width
		self.bit_offset = bit_offset
		self.domain_bits = domain_bits
		self.num_bits = num_bits


class DatetimeColumn(OrdinalColumn):
	"""Class for datetime column transformers	
	"""

	def __init__(self):
		super().__init__()
		self.col_type = "Datetime"

	def fit_by_metadata(self, metadata):
		self.col_name = metadata["name"]
		self.min = metadata["min"]
		self.max = metadata["max"]
		self.distribution_size = int(metadata["value_size"])
		self.distinct_values = np.unique(metadata["values"]).astype(np.object)
		self.dtype = np.datetime64
		date_range = pd.date_range(self.min, self.max, periods=self.distribution_size)
		for item in date_range:
			if len(self.distinct_values) >= self.distribution_size:
				break
			if metadata["type"] == "date":
				item = str(item.date())
			elif metadata["type"] == "time":
				item = str(item.time())
			else:
				item = str(item)
			if item not in self.distinct_values:
				self.distinct_values = np.append(self.distinct_values, item)
		self.distinct_values.sort()
		if "not_null" in metadata and metadata["not_null"] is False:
			self.distinct_values = np.append(self.distinct_values, "NULL")
			self.distribution_size += 1
		self.distinct_values = self.distinct_values.astype(np.object)
		assert len(self.distinct_values) == self.distribution_size
		self.dim = 1
		self.is_fit = True


class BinaryColumn(CategoricalColumn):
	"""Class for binary column transformers	
	"""

	def __init__(self):
		super().__init__()
		self.col_type = "Binary"

	def fit_by_instance(self, col_name, data_instance):
		self.col_name = col_name
		self.fit_data = data_instance
		self.get_distinct_values(data_instance)
		self.dim = max(1, int(np.ceil(np.log2(len(self.distinct_values)))))
		self.is_fit = True
	
	def fit_by_metadata(self, col_name, metadata):
		'''need to implement'''
		#raise NotImplementedError
		pass

	def transform(self, data_instance):
		allow_data = data_instance.copy()
		index_data = self.cat_to_index(allow_data)
		bit_data = self.index_to_bit(index_data)
		return bit_data.reshape(-1, self.dim)

	def inverse(self, data_instance):
		data_instance = np.round(data_instance)

		allow_data = data_instance.copy()
		index_data = self.bit_to_index(allow_data)
		cat_data = self.index_to_cat(index_data)

		return cat_data.reshape(-1, 1)

	def index_to_bit(self, index_data):
		bit_data = np.zeros((index_data.shape[0], self.dim))
		for i in range(index_data.shape[0]):
			bit_val = np.mod(np.right_shift(int(index_data[i]), list(reversed(np.arange(self.dim)))), 2)
			bit_data[i, :] = bit_val
		return bit_data 

	def bit_to_index(self, bit_data):
		assert bit_data.shape[1] == self.dim
		values = bit_data.astype(int)
		values = values * (2 ** np.array(list((reversed(np.arange(self.dim))))))
		values = np.sum(values, axis=1)
		return values


class OnehotColumn(CategoricalColumn):
	"""Class for one-hot column transformers	
	"""

	def __init__(self):
		super().__init__()
		self.col_type = "Onehot"

	def fit_by_instance(self, col_name, data_instance):
		self.col_name = col_name
		self.fit_data = data_instance
		self.get_distinct_values(data_instance)
		self.dim = self.distribution_size
		self.is_fit = True

	def fit_by_metadata(self, col_name, metadata):
		'''need to implement'''
		raise NotImplementedError

	def transform(self, data_instance):
		allow_data = data_instance.copy()
		index_data = self.cat_to_index(allow_data)
		onehot_data = self.index_to_onehot(index_data)
		return onehot_data
	
	def inverse(self, data_instance):
		assert exists(self.distinct_values)

		allow_data = data_instance.copy()
		index_data = self.onehot_to_index(allow_data)
		cat_data = self.index_to_cat(index_data)

		return cat_data.reshape(-1,1)

	def index_to_onehot(self, index_data):
		onehot_data = np.zeros([index_data.shape[0], self.dim])
		idx = np.arange(len(index_data))
		onehot_data[idx, index_data.astype(int).reshape(1, -1)] = 1
		return onehot_data 	

	def onehot_to_index(self, onehot_data):
		assert onehot_data.shape[1] == self.dim 
		index_data = np.argmax(onehot_data, axis=1)
		return index_data


def find_closest_value(element, array):
	return min(array, key=lambda x: abs(x - element))
