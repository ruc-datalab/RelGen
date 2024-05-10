from typing import Dict
import pandas as pd
import torch

from relgen.data.metadata import Metadata
from relgen.data.column import Column, DiscreteColumn, OrdinalColumn
from relgen.data.table import Table


class Dataset:
    def __init__(self, metadata: Metadata, num_column: Column = DiscreteColumn(), cat_column: Column = OrdinalColumn()):
        self.metadata = metadata
        self.num_column = num_column
        self.cat_column = cat_column
        self.tables = {}
        for table in metadata.tables.keys():
            self.tables[table] = Table(name=table, num_column=num_column, cat_column=cat_column)
        self.is_fitted = False

    def fit(self, data: Dict[str, pd.DataFrame]):
        self.is_fitted = True
        for table_name, table_data in data.items():
            self.tables[table_name].fit(table_data)

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
        transformed_data = {}
        for table_name, table_data in data.items():
            transformed_data[table_name] = self.tables[table_name].transform(data[table_name])
        return transformed_data

    def inverse(self, data: Dict[str, torch.Tensor]) -> Dict[str, pd.DataFrame]:
        inverse_data = {}
        for table_name, table_data in data.items():
            inverse_data[table_name] = self.tables[table_name].inverse(data[table_name])
        return inverse_data

    def adapt2metadata(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        results = data
        for table_name, table_data in data.items():
            if "primary_key" in self.metadata.tables[table_name].keys():
                primary_key = self.metadata.tables[table_name]["primary_key"]
                if results[table_name][primary_key].duplicated().any():
                    results[table_name][primary_key] = range(len(results[table_name]))
        return results

    def join_data(self, data: Dict[str, pd.DataFrame]):
        join_table_data = None
        for relationship in self.metadata.sorted_relationships:
            child_table_name = relationship["child_table_name"]
            child_table_data = data[child_table_name]
            if join_table_data is None:
                join_table_data = child_table_data
            else:
                join_table_data = pd.merge(join_table_data, child_table_data,
                                           left_on=relationship["parent_foreign_key"],
                                           right_on=relationship["child_primary_key"])
        return join_table_data
