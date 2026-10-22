import torch

def count_params(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    return total

sdf_path = "vkils59_custom_task/trainings/training_sdf_20260903-154542/best_inr.pt"
flow_path = "vkils59_custom_task/trainings/training_result_20260903-180038/ms_fourier_features_nl6_np16000_epochs1000_lr0.001_ldim10.pt"

n_sdf = count_params(sdf_path)
n_flow = count_params(flow_path)
print(f"SDF encoder: {n_sdf:,}")
print(f"Flow model:  {n_flow:,}")
print(f"Total:       {n_sdf + n_flow:,}")


ckpt = torch.load(sdf_path, map_location="cpu", weights_only=False)
print(type(ckpt), ckpt.keys() if isinstance(ckpt, dict) else None)


import torch

ckpt = torch.load(flow_path, map_location="cpu", weights_only=False)
for k, v in ckpt.items():
    print(k, type(v))




def count_tensors(obj):
    if hasattr(obj, "numel"):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(count_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_tensors(v) for v in obj)
    return 0

def count_params(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    total = 0
    for key in ("inr_in", "alpha_in"):
        if key in ckpt:
            n = count_tensors(ckpt[key])
            print(f"  {key}: {n:,}")
            total += n
    return total

n_sdf = count_params(sdf_path)
n_flow = count_params(flow_path)
print(f"SDF encoder total: {n_sdf:,}")
print(f"Flow model total:  {n_flow:,}")
print(f"Grand total:       {n_sdf + n_flow:,}")