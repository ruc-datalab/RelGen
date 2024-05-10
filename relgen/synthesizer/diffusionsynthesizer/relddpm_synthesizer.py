import time
from copy import deepcopy
from typing import Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from relgen.utils import constant
from relgen.utils.enum_type import SynthesisMethod
from relgen.data.data_processor import join_and_add_virtual_column, group_and_merge
from relgen.data.utils import prepare_fast_dataloader
from relgen.data.table import Table
from relgen.data.dataset import Dataset
from relgen.model.diffusionmodel import modules, DiffusionModel, GaussianDiffusion
from relgen.synthesizer.diffusionsynthesizer import DiffusionSynthesizer
from relgen.synthesizer.diffusionsynthesizer.resample import create_named_schedule_sampler


class RelDDPMSynthesizer(DiffusionSynthesizer):
    def __init__(self, dataset: Dataset, method: SynthesisMethod = SynthesisMethod.SINGLE_MODEL):
        super(RelDDPMSynthesizer, self).__init__(dataset)
        if method == SynthesisMethod.MULTI_MODEL:
            raise ValueError("RelDDPMSynthesizer can not use multi model method yet")
        self.method = method
        self.models = {}
        self.controllers = {}

    def fit(self, data: Dict[str, pd.DataFrame], device=torch.device("cpu"), condition=None, epochs: int = 30000, controller_steps: int = 2000,
            batch_size: int = 4096, controller_batch_size: int = 4096, verbose: bool = True, show_progress: bool = False):
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
                    parent_table_data, child_table_data, join_table_data = join_and_add_virtual_column(parent_table, parent_table_data, relationship["parent_foreign_key"], child_table, child_table_data, relationship["child_primary_key"])
                    join_table_name = f"{parent_table_name}{constant.JOIN_OPERATOR}{child_table_name}"
                    join_table = Table(join_table_name, num_column=self.dataset.num_column,
                                       cat_column=self.dataset.cat_column)
                    join_table.fit(child_table_data)
                    self.dataset.tables[join_table_name] = join_table
                    model = self._train_model(join_table, child_table_data, device, epochs=epochs, bs=batch_size)
                    self.models[child_table_name] = model

                    controller = self._train_controller(parent_table, parent_table_data, join_table, child_table_data,
                                                        model, device, steps=controller_steps, batch_size=controller_batch_size)
                    self.controllers[f"{parent_table_name}{constant.JOIN_OPERATOR}{child_table_name}"] = controller
                else:
                    model = self._train_model(child_table, child_table_data, device, epochs=epochs, bs=batch_size)
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
            model = self._train_model(join_table, join_table_data, device, epochs=epochs, bs=batch_size)
            self.models[join_table_name] = model

    def _train_model(self, table: Table, data: pd.DataFrame, device, d_hidden=[512, 1024, 1024, 512],
                     num_timesteps=1000, epochs=30000, lr=0.0018, drop_out=0.0, bs=4096) -> GaussianDiffusion:
        train_x = table.transform(data)
        train_x = torch.from_numpy(train_x).float()
        model = modules.MLPDiffusion(train_x.shape[1], d_hidden, drop_out)
        model.to(device)
        print("Model Initialization")

        diffuser = GaussianDiffusion(train_x.shape[1], model, device=device, num_timesteps=num_timesteps)
        diffuser.to(device)
        diffuser.train()
        print("Diffusion Initialization")

        ds = [train_x]
        dl = prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)

        trainer = DiffusionTrainer(diffuser, dl, lr, 0.0, epochs, save_path=None, device=device)
        train_sta = time.time()
        trainer.run_loop()
        train_end = time.time()
        print(f'training time: {train_end - train_sta}')
        return diffuser

    def _train_controller(self, condition_table: Table, condition_data: pd.DataFrame, synthetic_table: Table,
                          synthetic_data: pd.DataFrame, diffuser: GaussianDiffusion, device,
                          d_hidden=[128, 128], lr=0.0015, steps=2000, batch_size=4096) -> nn.Module:
        # condition_table = du.merge_table(condition_table)
        # synthetic_table = du.merge_table(synthetic_table)

        condition_data = condition_table.transform(condition_data)
        synthetic_data = synthetic_table.transform(synthetic_data)

        train_cond_norm = torch.as_tensor(condition_data).float()
        train_data_norm = torch.as_tensor(synthetic_data).float()

        diffuser.to(device)
        diffuser.variables_to_device(device)
        diffuser.eval()

        cond_encoder = modules.MLPEncoder(train_cond_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=False)
        data_encoder = modules.MLPEncoder(train_data_norm.shape[1], d_hidden, 128, 0.0, 128, t_in=True)
        controller = modules.CondScorer(cond_encoder, data_encoder)
        controller.to(device)

        ds = [train_cond_norm, train_data_norm]
        dl = prepare_fast_dataloader(ds, batch_size=batch_size, shuffle=True)

        schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

        opt = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.0)

        sta = time.time()
        losses = []
        for step in range(steps):
            c, x = next(dl)
            c = c.to(device)
            x = x.to(device)

            t, _ = schedule_sampler.sample(len(x), device)
            x_t = diffuser.gaussian_q_sample(x, t)
            # c_t = diffuser.gaussian_q_sample(c, t)

            logits_c, logits_x = controller(c, x_t, t)
            labels = np.arange(logits_c.shape[0])
            labels = torch.as_tensor(labels).to(device)

            loss_1 = F.cross_entropy(logits_c, labels)
            loss_2 = F.cross_entropy(logits_x, labels)
            loss = (loss_1 + loss_2) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()

            set_anneal_lr(opt, lr, step, steps)

            if (step + 1) % 1000 == 0 or step == 0:
                print(f'Step {step + 1}/{steps} : Loss {loss.data}, loss1 {loss_1.data}, loss2 {loss_2.data}')
                losses.append(loss.detach().cpu().numpy())
        end = time.time()
        train_elapse = end - sta
        print(f"training time: {train_elapse}")
        return controller

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
                    model = self.models[child_table_name]
                    controller = self.controllers[f"{parent_table_name}{constant.JOIN_OPERATOR}{child_table_name}"]
                    conditional_data, sampled_data = self._conditional_sample(parent_table, child_table, parent_table_data, model, controller, device)
                    conditional_data, sampled_data = group_and_merge(parent_table, parent_table_data,
                                                                     relationship["parent_foreign_key"], child_table,
                                                                     sampled_data, relationship["child_primary_key"])
                    sampled_results[parent_table_name] = conditional_data
                    sampled_results[child_table_name] = sampled_data
                else:
                    sampled_data = self.models[child_table_name].sample(child_table.cardinality)
                    sampled_data = child_table.inverse(sampled_data)
                    sampled_results[child_table_name] = sampled_data
        else:
            join_table_name = f"{constant.JOIN_OPERATOR}".join(
                [relationship["child_table_name"] for relationship in sorted_relationships])
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
                    sampled_data = full_sampled_data[
                        child_table.col_names + [f"{constant.WEIGHT_VIRTUAL_COLUMN}_{child_table.name}"]]
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

    def _conditional_sample(self, condition_table: Table, synthetic_table: Table, condition_data: pd.DataFrame, diffuser: GaussianDiffusion, controller: nn.Module, device=torch.device("cpu"),
                            scale_factor=25, bs=100000):
        # condition_table = du.merge_table(condition_table)
        # synthetic_table = du.merge_table(synthetic_table)
        # condition_data = condition_data[condition_table.raw_columns]

        test_cond_norm = condition_table.transform(condition_data)
        test_cond_norm = torch.as_tensor(test_cond_norm).float()

        diffuser.eval()
        diffuser.to(device)
        diffuser.variables_to_device(device)

        controller.eval()
        controller.to(device)

        cond_fn = get_cond_fn(controller, scale_factor)

        sample_index = np.arange(len(test_cond_norm))
        sample_data = np.zeros([len(test_cond_norm), sum(synthetic_table.col_dims)])

        while len(sample_index) > 0:
            cond_input = test_cond_norm[sample_index, :]
            control_tools = (cond_input, cond_fn)

            sample = diffuser.batch_sample(len(cond_input), batch_size=bs, control_tools=control_tools)
            sample = sample.cpu().numpy()
            sample = synthetic_table.inverse(sample)
            sample = sample.values

            allow_index, reject_index = self.reject_sample(synthetic_table, sample)
            sample_allow_index = sample_index[allow_index] if len(allow_index) > 0 else []
            sample_reject_index = sample_index[reject_index] if len(reject_index) > 0 else []

            if len(sample_allow_index) > 0:
                sample_data[sample_allow_index, :] = sample[allow_index, :]
            sample_index = sample_reject_index

        # sample_data = synthetic_table.ReverseToCat(sample_data)
        # sample_data = pd.DataFrame(sample_data, columns=synthetic_table.columns)
        # sample_data = synthetic_table.ReOrderColumns(sample_data)
        sample_data = synthetic_table.inverse(sample_data)
        return condition_data, sample_data

    def reject_sample(self, table: Table, sample):
        all_index = set(range(sample.shape[0]))
        allow_index = set(range(sample.shape[0]))
        for i, col in enumerate(table.columns):
            allow_index = allow_index & set(np.where(sample[:, i] < len(col.distinct_values))[0])
            allow_index = allow_index & set(np.where(sample[:, i] >= 0)[0])
        reject_index = all_index - allow_index
        allow_index = np.array(list(allow_index))
        reject_index = np.array(list(reject_index))
        return allow_index, reject_index

    def save(self, save_path: str):
        state = {
            "models": {},
            "controllers": {}
        }
        for table, model in self.models.items():
            state["models"][table] = model.state_dict()
        for table, controller in self.controllers.items():
            state["controllers"][table] = controller.state_dict()
        torch.save(state, save_path)

    def load(self, load_path: str, device=torch.device("cpu")):
        checkpoint = torch.load(load_path, map_location=device)
        for table in self.models.keys():
            self.models[table].load_state_dict(checkpoint["models"][table])
        for table in self.controllers.keys():
            self.controllers[table].load_state_dict(checkpoint["controllers"][table])


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


class DiffusionTrainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, save_path=None, num_checkpoints=1,
                 device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.is_cond = self.diffusion._denoise_fn.is_cond
        if self.is_cond:
            print("Conditional Training!")
        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.step_per_check = steps // num_checkpoints
        self.save_path = save_path

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, cond=None, epsilon=None):
        x = x.to(self.device)
        if self.is_cond and cond is not None:
            cond = cond.to(self.device)
        self.optimizer.zero_grad()

        loss = self.diffusion.calculate_loss(x, cond)
        loss.backward()
        # for p in self.diffusion.parameters():
        # #     print(p)
        # #     print(p.grad)
        #      p.grad.clamp_(-0.1, 0.1)
        self.optimizer.step()

        return loss

    def run_loop(self):
        step = 0
        curr_loss = 0.0

        curr_count = 0
        train_start = time.time()
        while step < self.steps:
            if self.is_cond:
                x, cond = next(self.train_iter)
            else:
                x = next(self.train_iter)[0]
                cond = None

            batch_loss = self._run_step(x, cond)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                loss = np.around(curr_loss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, loss]
                curr_count = 0
                curr_loss = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
        train_end = time.time()

        self.loss_history.loc[len(self.loss_history)] = [step, train_end - train_start]


def set_anneal_lr(opt, init_lr, step, all_steps):
    frac_done = step / all_steps
    lr = init_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def get_cond_fn(controller, scale_factor):
    def cond_fn(c, x, t):
        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)

            x_features = controller.data_encoder(x_in, t)
            c_features = controller.cond_encoder(c, t)

            c_features = c_features / c_features.norm(dim=1, keepdim=True)
            x_features = x_features / x_features.norm(dim=1, keepdim=True)

            logits = torch.einsum("bi,bi->b", [c_features, x_features])
            gradients = torch.autograd.grad(torch.sum(logits), x_in)[0] * scale_factor
            return gradients

    return cond_fn
