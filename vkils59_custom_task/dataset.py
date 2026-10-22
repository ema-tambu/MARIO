import os
import json
import pickle
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Dataset, Data


def load_manifest(root_path):
    manifest_path = os.path.join(root_path, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return manifest["train_samples"], manifest["test_samples"]


def load_vkils59(root_path, train=True):
    """
    Mimics the (data_list, data_names) signature returned by airfrans.dataset.load,
    but reads our own pkl files + manifest.json instead.
    """
    train_samples, test_samples = load_manifest(root_path)
    sample_names = train_samples if train else test_samples

    data_list = []
    data_names = []
    for fname in sample_names:
        fpath = os.path.join(root_path, fname)
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        params = d["params"]
        sample = {
            "pos": np.asarray(d["X_original"], dtype=np.float64),
            "sdf": np.asarray(d["sdf"], dtype=np.float64),
            "fields": np.asarray(d["fields"], dtype=np.float64),
            "angle_in": float(params[7]),
            "mach_out": float(params[8]),
        }
        data_list.append(sample)
        data_names.append(os.path.splitext(fname)[0])

    return data_list, data_names


def make_bl_layer(sdf, thickness: float = 0.03, decay: Literal['linear', 'exponential', 'squared'] = 'squared'):
    if not (0 < thickness <= 1):
        raise ValueError('Thickness should be in (0, 1].')

    def threshold(var, value):
        mask = var > value
        var = var.copy()
        var[~mask] = 0
        if mask.any():
            var[mask] = (var[mask] - value) / (var[mask].max() - value)
        return var

    df = sdf.max() - sdf
    df = df / df.max()

    if decay == 'squared':
        return threshold(df, 1 - thickness) ** 2
    elif decay == 'linear':
        return threshold(df, 1 - thickness)
    elif decay == 'exponential':
        return (np.exp(threshold(df, 1 - thickness)) - 1) / (np.exp(1) - 1)
    else:
        raise ValueError(f"Unknown decay type: {decay}")


class SDFDataset:
    def __init__(self, data_list, data_names, is_train=True, coef_norm=None, num_points=None):
        self.data_list = data_list
        self.data_names = data_names
        self.num_points = num_points
        self.is_train = is_train

        if is_train:
            print("Computing normalization parameters from training data...")
            self.compute_norm_params(data_list)
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']

        print("Processing dataset...")
        self.processed_dataset = self.process_data(data_list)

    def compute_norm_params(self, data_list):
        self.coef_norm = {}

        all_targets = np.concatenate([d['sdf'] for d in data_list])
        self.coef_norm['mean'] = float(all_targets.mean())
        self.coef_norm['std'] = float(all_targets.std())

        all_positions = np.concatenate([d['pos'] for d in data_list], axis=0)
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm

        print("\nNormalization parameters computed:")
        print(f"Target mean: {self.coef_norm['mean']:.4f}")
        print(f"Target std: {self.coef_norm['std']:.4f}")
        return self.coef_norm

    def process_data(self, dataset_list):
        dataset = []
        for d in dataset_list:
            pos = d['pos']
            pos_range = self.pos_norm['max'] - self.pos_norm['min']
            pos_n = 2 * (pos - self.pos_norm['min']) / pos_range - 1

            sdf = d['sdf']
            sdf_n = (sdf - self.coef_norm['mean']) / self.coef_norm['std']

            data_entry = Data(
                pos=torch.tensor(pos_n, dtype=torch.float),
                input=torch.tensor(pos_n, dtype=torch.float),
                output=torch.tensor(sdf_n[:, None], dtype=torch.float),
            )
            dataset.append(data_entry)

        if self.num_points is not None:
            dataset = subsample_dataset(dataset, self.num_points)

        return dataset


class VKILS59FlowDataset(Dataset):
    def __init__(
        self,
        data_list,
        data_names,
        geom_latents,
        mode='train',
        coef_norm=None,
        num_points=None,
        idx_list=None,
        transform=None,
        pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.data_list = data_list
        self.data_names = data_names
        self._raw_geom_latents = np.array(geom_latents)

        total = len(self.data_list)
        self.idx_list = list(range(total)) if idx_list is None else list(idx_list)
        self.geom_latents = self._raw_geom_latents[self.idx_list]
        self.mode = mode
        self.num_points = num_points

        self.bl_thickness = 0.03
        self.bl_decay = 'squared'

        self.output_fields = ['Mach', 'Turbulent-viscosity']

        if mode == 'train':
            self.coef_norm = self.compute_norm_params()
        else:
            if coef_norm is None:
                raise ValueError("coef_norm required for non-train modes")
            self.coef_norm = coef_norm

        self.processed_data_list = self.process_dataset()

    def compute_norm_params(self):
        pos_all, sdf_all, bl_all = [], [], []
        output_fields_all = {0: [], 1: []}
        cond_all = []

        for i, idx in enumerate(self.idx_list):
            d = self.data_list[idx]
            pos = d['pos']
            sdf = d['sdf']
            bl_mask = make_bl_layer(sdf, thickness=self.bl_thickness, decay=self.bl_decay)
            fields = d['fields']

            pos_all.append(pos)
            sdf_all.append(sdf)
            bl_all.append(bl_mask)
            output_fields_all[0].append(fields[:, 0])
            output_fields_all[1].append(fields[:, 1])

            cond = np.concatenate([
                self.geom_latents[i],
                np.array([d['angle_in'], d['mach_out']])
            ])
            cond_all.append(cond)

        pos_all = np.vstack(pos_all)
        pos_min = pos_all.min(axis=0)
        pos_max = pos_all.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0

        sdf_all = np.hstack(sdf_all)
        sdf_min, sdf_max = sdf_all.min(), sdf_all.max()
        sdf_range = sdf_max - sdf_min
        sdf_range = 1.0 if sdf_range == 0 else sdf_range

        bl_all = np.hstack(bl_all)
        bl_min, bl_max = bl_all.min(), bl_all.max()
        bl_range = bl_max - bl_min
        bl_range = 1.0 if bl_range == 0 else bl_range

        output_means, output_stds = {}, {}
        for i, field_name in enumerate(self.output_fields):
            field_data = np.hstack(output_fields_all[i])
            output_means[field_name] = float(field_data.mean())
            output_stds[field_name] = float(field_data.std()) or 1.0

        cond_all = np.vstack(cond_all)
        cond_mean = cond_all.mean(axis=0)
        cond_std = cond_all.std(axis=0)
        cond_std[cond_std == 0] = 1.0

        return {
            'pos': {'min': pos_min, 'max': pos_max, 'range': pos_range},
            'sdf': {'min': sdf_min, 'max': sdf_max, 'range': sdf_range},
            'bl': {'min': bl_min, 'max': bl_max, 'range': bl_range},
            'output': {'mean': output_means, 'std': output_stds},
            'cond': {'mean': cond_mean, 'std': cond_std}
        }

    def process_dataset(self):
        processed_data_list = []
        c = self.coef_norm

        for i, idx in enumerate(self.idx_list):
            d = self.data_list[idx]
            pos = d['pos']
            sdf = d['sdf']
            fields = d['fields']

            if self.num_points and pos.shape[0] > self.num_points:
                sel = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos, sdf, fields = pos[sel], sdf[sel], fields[sel]

            bl_mask = make_bl_layer(sdf, thickness=self.bl_thickness, decay=self.bl_decay)

            pos_n = 2 * (pos - c['pos']['min']) / c['pos']['range'] - 1
            sdf_n = 2 * ((sdf - c['sdf']['min']) / c['sdf']['range']) - 1
            bl_n = 2 * ((bl_mask - c['bl']['min']) / c['bl']['range']) - 1

            input_feats = np.concatenate([pos_n, sdf_n[:, None], bl_n[:, None]], axis=1)
            node_kwargs = {
                'input': torch.tensor(input_feats, dtype=torch.float),
                'pos': torch.tensor(pos_n, dtype=torch.float),
            }

            output_data = []
            for j, field_name in enumerate(self.output_fields):
                mean_val = c['output']['mean'][field_name]
                std_val = c['output']['std'][field_name]
                field_norm = (fields[:, j] - mean_val) / std_val
                output_data.append(field_norm[:, None])
            node_kwargs['output'] = torch.tensor(np.hstack(output_data), dtype=torch.float)

            cond = np.concatenate([self.geom_latents[i], [d['angle_in'], d['mach_out']]])
            cm, cs = c['cond']['mean'], c['cond']['std']
            cond_n = (cond - cm) / cs
            node_kwargs['cond'] = torch.tensor(cond_n, dtype=torch.float).unsqueeze(0)

            processed_data_list.append(Data(**node_kwargs))

        return processed_data_list

    def len(self):
        return len(self.processed_data_list)

    def get(self, idx):
        return self.processed_data_list[idx]


def subsample_dataset(dataset, num_points, seed=None):
    if seed is not None:
        np.random.seed(seed)

    out = []
    for data in dataset:
        N = data.input.size(0)
        if N > num_points:
            idx = np.random.choice(N, num_points, replace=False)
            idx = torch.from_numpy(idx).long()

            inp_sub = data.input[idx]
            pos_sub = data.pos[idx]
            out_sub = data.output[idx] if hasattr(data, 'output') else None

            new_data = Data(input=inp_sub, pos=pos_sub)
            if out_sub is not None:
                new_data.output = out_sub
            if hasattr(data, 'cond'):
                new_data.cond = data.cond
        else:
            new_data = data

        out.append(new_data)
    return out
