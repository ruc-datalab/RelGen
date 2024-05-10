import time
from typing import Dict
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer

from relgen.utils import constant
from relgen.utils.enum_type import SynthesisMethod
from relgen.data.data_processor import join_and_add_virtual_column, group_and_merge
from relgen.data.table import Table
from relgen.data.dataset import Dataset
from relgen.model.armodel import MADE
from relgen.synthesizer.arsynthesizer import ARSynthesizer


class MADESynthesizer(ARSynthesizer):
    def __init__(self, dataset: Dataset, models: Dict[str, MADE] = {}, method: SynthesisMethod = SynthesisMethod.MULTI_MODEL):
        super(MADESynthesizer, self).__init__(dataset)
        self.models = models
        self.method = method

    def fit(self, data: Dict[str, pd.DataFrame], device=torch.device("cpu"), condition=None, epochs: int = 100, batch_size: int = 1024, verbose: bool = True, show_progress: bool = False):
        sorted_relationships = self.dataset.metadata.sorted_relationships
        if self.method == SynthesisMethod.MULTI_MODEL:
            for relationship in sorted_relationships:
                child_table_name = relationship["child_table_name"]
                child_table_data = data[child_table_name]
                child_table = self.dataset.tables[child_table_name]
                if "parent_table_name" in relationship.keys():
                    parent_table_name = relationship["parent_table_name"]
                    parent_table_data = data[parent_table_name]
                    parent_table = self.dataset.tables[parent_table_name]
                    _, _, join_table_data = join_and_add_virtual_column(parent_table, parent_table_data,
                                                                        relationship["parent_foreign_key"], child_table,
                                                                        child_table_data,
                                                                        relationship["child_primary_key"])
                    join_table_name = f"{parent_table_name}{constant.JOIN_OPERATOR}{child_table_name}"
                    join_table = Table(join_table_name, num_column=self.dataset.num_column,
                                       cat_column=self.dataset.cat_column)
                    join_table.fit(join_table_data)
                    self.dataset.tables[join_table_name] = join_table
                    model = MADE(join_table)
                    optimizer = torch.optim.Adam(model.parameters())
                    self._train_model(join_table, join_table_data, model, optimizer, device, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose, show_progress=show_progress)
                    self.models[join_table_name] = model
                else:
                    model = MADE(child_table)
                    optimizer = torch.optim.Adam(model.parameters())
                    self._train_model(child_table, child_table_data, model, optimizer, device, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose, show_progress=show_progress)
                    self.models[child_table_name] = model
        else:
            join_table_name = None
            join_table_data = None
            for relationship in sorted_relationships:
                child_table_name = relationship["child_table_name"]
                child_table_data = data[child_table_name]
                child_table = self.dataset.tables[child_table_name]
                if join_table_data is None:
                    join_table_name = child_table_name
                    join_table_data = child_table_data
                else:
                    parent_table_name = relationship["parent_table_name"]
                    parent_table_data = data[parent_table_name]
                    parent_table = self.dataset.tables[parent_table_name]
                    join_table_name += f"{constant.JOIN_OPERATOR}{child_table_name}"
                    _, _, join_table_data = join_and_add_virtual_column(parent_table, join_table_data,
                                                                        relationship["parent_foreign_key"], child_table,
                                                                        child_table_data,
                                                                        relationship["child_primary_key"])
            join_table = Table(join_table_name, num_column=self.dataset.num_column, cat_column=self.dataset.cat_column)
            join_table.fit(join_table_data)
            self.dataset.tables[join_table_name] = join_table
            model = MADE(join_table)
            optimizer = torch.optim.Adam(model.parameters())
            self._train_model(join_table, join_table_data, model, optimizer, device, epochs=epochs,
                              batch_size=batch_size, verbose=verbose, show_progress=show_progress)
            self.models[join_table_name] = model

    def _train_model(self, table: Table, data: pd.DataFrame, model: MADE, optimizer: Optimizer, device=torch.device("cpu"), condition=None, epochs: int = 100, batch_size: int = 1024, verbose: bool = True, show_progress: bool = False):
        train_data = table.transform(data)
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.logger.info(f"table {table.name} start training")
        start_time = time.time()
        for epoch_idx in range(epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(dataloader, model, optimizer, device, condition, show_progress)
            epoch_end_time = time.time()
            if verbose:
                self.logger.info("epoch %d: train loss %.3f, time cost %.3fs" % (epoch_idx, train_loss, epoch_end_time - epoch_start_time))
        end_time = time.time()
        self.logger.info("table %s training completed, time cost %.3fs" % (table.name, end_time - start_time))

    def _train_epoch(self, dataloader: DataLoader, model: MADE, optimizer: Optimizer, device=torch.device("cpu"), condition=None, show_progress: bool = False) -> float:
        iter_data = (
            tqdm(dataloader)
            if show_progress
            else dataloader
        )
        model.train()
        total_loss = 0.
        for batch_idx, batch_data in enumerate(iter_data):
            if hasattr(model, 'update_masks'):
                model.update_masks()
            batch_data = batch_data.float().to(device)
            output = model(batch_data)
            loss = model.calculate_loss(output, batch_data)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
        return total_loss / len(iter_data)

    def sample(self, condition=None, device=torch.device("cpu")) -> Dict[str, pd.DataFrame]:
        sorted_relationships = self.dataset.metadata.sorted_relationships
        sampled_results = {}
        if self.method == SynthesisMethod.MULTI_MODEL:
            for relationship in sorted_relationships:
                child_table_name = relationship["child_table_name"]
                child_table = self.dataset.tables[child_table_name]
                if "parent_table_name" in relationship.keys():
                    parent_table_name = relationship["parent_table_name"]
                    parent_table = self.dataset.tables[parent_table_name]
                    parent_table_data = sampled_results[parent_table_name]
                    join_table_name = f"{parent_table_name}{constant.JOIN_OPERATOR}{child_table_name}"
                    join_table = self.dataset.tables[join_table_name]
                    conditional_data = parent_table.transform(parent_table_data)
                    conditional_data = torch.tensor(conditional_data)
                    sampled_data = self.models[join_table_name].sample(len(parent_table_data), conditional_data, device)
                    sampled_data = join_table.inverse(sampled_data)
                    sampled_data = sampled_data[child_table.col_names + [f"{constant.WEIGHT_VIRTUAL_COLUMN}_{child_table.name}"]]
                    conditional_data, sampled_data = group_and_merge(parent_table, parent_table_data,
                                                                     relationship["parent_foreign_key"], child_table,
                                                                     sampled_data, relationship["child_primary_key"])
                    sampled_results[parent_table_name] = conditional_data
                    sampled_results[child_table_name] = sampled_data
                else:
                    sampled_data = self.models[child_table_name].sample(child_table.cardinality, device=device)
                    sampled_data = child_table.inverse(sampled_data)
                    sampled_results[child_table_name] = sampled_data
        else:
            join_table_name = f"{constant.JOIN_OPERATOR}".join([relationship["child_table_name"] for relationship in sorted_relationships])
            root_table_name = sorted_relationships[0]["child_table_name"]
            full_sampled_data = self.models[join_table_name].sample(self.dataset.tables[root_table_name].cardinality)
            full_sampled_data = self.dataset.tables[join_table_name].inverse(full_sampled_data)
            for relationship in sorted_relationships:
                child_table_name = relationship["child_table_name"]
                child_table = self.dataset.tables[child_table_name]
                if "parent_table_name" in relationship.keys():
                    parent_table_name = relationship["parent_table_name"]
                    parent_table = self.dataset.tables[parent_table_name]
                    parent_table_data = sampled_results[parent_table_name]
                    sampled_data = full_sampled_data[child_table.col_names + [f"{constant.WEIGHT_VIRTUAL_COLUMN}_{child_table.name}"]]
                    conditional_data, sampled_data = group_and_merge(parent_table, parent_table_data,
                                                                     relationship["parent_foreign_key"], child_table,
                                                                     sampled_data, relationship["child_primary_key"])
                    sampled_results[parent_table_name] = conditional_data
                    sampled_results[child_table_name] = sampled_data
                else:
                    sampled_data = full_sampled_data[child_table.col_names]
                    sampled_results[child_table_name] = sampled_data
        sampled_results = self.dataset.adapt2metadata(sampled_results)
        return sampled_results

    def save(self, save_path: str):
        state = {
            "models": {}
        }
        for table, model in self.models.items():
            state["models"][table] = model.state_dict()
        torch.save(state, save_path)

    def load(self, load_path: str, device=torch.device("cpu")):
        checkpoint = torch.load(load_path, map_location=device)
        for table in self.models.keys():
            self.models[table].load_state_dict(checkpoint["models"][table])
