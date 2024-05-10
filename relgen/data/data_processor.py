import warnings
import pandas as pd

from relgen.utils import constant
from relgen.data.table import Table

warnings.filterwarnings("ignore")


def join_and_add_virtual_column(cond_table: Table, cond_data: pd.DataFrame, cond_join_key: str, table: Table, data: pd.DataFrame, join_key: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    pd.set_option('mode.chained_assignment', None)
    weight_col_name = f"{constant.WEIGHT_VIRTUAL_COLUMN}_{table.name}"
    merged_data = pd.merge(cond_data, data, left_on=cond_join_key, right_on=join_key)
    merged_data[weight_col_name] = 1.0
    # merged_data[indicator_col_name] = 1
    for index, row in merged_data.iterrows():
        fanout_indicator = len(cond_data[cond_data[cond_join_key] == merged_data.at[index, join_key]])
        if fanout_indicator != 0:
            merged_data.at[index, weight_col_name] = 1.0 / fanout_indicator
    # if merged_data.at[index, cond_join_key] not in data[join_key].values:
    # 	merged_data.at[index, indicator_col_name] = 0
    tmp_cond_data = merged_data[cond_data.columns]
    tmp_data = merged_data[data.columns]
    tmp_data[weight_col_name] = merged_data[weight_col_name]
    # if cond_join_key in tmp_cond_data.columns:
    #     tmp_cond_data = tmp_cond_data.drop(cond_join_key, axis=1)
    # if join_key in tmp_data.columns:
    #     tmp_data = tmp_data.drop(join_key, axis=1)
    # if cond_join_key in merged_data.columns:
    #     merged_data = merged_data.drop(cond_join_key, axis=1)
    # if join_key in merged_data.columns:
    #     merged_data = merged_data.drop(join_key, axis=1)
    return tmp_cond_data, tmp_data, merged_data


def group_and_merge(cond_table: Table, cond_data: pd.DataFrame, cond_join_key: str, table: Table, data: pd.DataFrame, join_key: str) -> (pd.DataFrame, pd.DataFrame):
    assert len(cond_data) == len(data)
    weight_col_name = f"{constant.WEIGHT_VIRTUAL_COLUMN}_{table.name}"
    ret_cond_data = cond_data.copy()
    ret_data = pd.DataFrame(columns=data.columns)
    counter = 0
    grouped = data.groupby(list(data.columns))
    for _, group in grouped:
        row = data.iloc[group.index.tolist()[0]]
        index_list = []
        weight_sum = 0.0
        for i in group.index.tolist():
            index_list.append(i)
            weight_sum += data.loc[i, weight_col_name]
            if weight_sum >= 1.0:
                ret_cond_data.loc[index_list, cond_join_key] = counter
                row[join_key] = counter
                # ret_data = ret_data.append(row, ignore_index=True)
                ret_data = pd.concat([ret_data, pd.DataFrame([row])], ignore_index=True)
                counter += 1
                index_list = []
                weight_sum = 0.0
        if len(index_list) > 0:
            index_list = [i for i in index_list if i < len(ret_cond_data)]
            ret_cond_data.loc[index_list, cond_join_key] = counter
            row[join_key] = counter
            # ret_data = ret_data.append(row, ignore_index=True)
            ret_data = pd.concat([ret_data, pd.DataFrame([row])], ignore_index=True)
            counter += 1
    ret_data = ret_data.drop(weight_col_name, axis=1)
    return ret_cond_data, ret_data
