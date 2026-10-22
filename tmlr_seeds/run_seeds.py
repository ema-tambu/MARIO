"""
Seeded MARIO reruns for the TMLR revision (AirfRANS full/scarce, VKI-LS59), two stages per run:
SDF geometry encoder (train_sdf.py) then flow model (train.py) on the latents of that encoder.

Each run uses the exact resolved config of the run reported in the submitted paper, plus +seed=<s>.
The output folders are identified by the seed/task stored in their config.yaml (two lanes share folders).

usage: CUDA_VISIBLE_DEVICES=0 python tmlr_seeds/run_seeds.py full:0 full:1 vki:0 ...
"""
import csv, os, shutil, socket, subprocess, sys, time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "tmlr_seeds" / "configs"
LOG = ROOT / "tmlr_seeds" / ("smoke_runs.csv" if os.environ.get("MARIO_SMOKE") else "runs.csv")
PY = sys.executable

REF = {  # task -> (folder, reference SDF run, reference flow run) used in the submitted paper
    "full": ("airfrans_task", "training_sdf_20260902-111728", "training_result_20260902-142820"),
    "scarce": ("airfrans_task", "training_sdf_20260902-111725", "training_result_20260902-142504"),
    "vki": ("vkils59_custom_task", "training_sdf_20260903-154542", "training_result_20260903-180038"),
}
LATENT = {"full": ("full_train_8.npz", "full_test_8.npz"), "scarce": ("scarce_train_8.npz", "scarce_test_8.npz"),
          "vki": ("vkils59_train_8.npz", "vkils59_test_8.npz")}


def prepare_configs(task):
    folder, sdf, flow = REF[task]
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for kind, run in (("sdf", sdf), ("flow", flow)):
        cfg = yaml.safe_load(open(ROOT / folder / "trainings" / run / "config.yaml"))
        cfg.pop("seed", None)
        with open(CFG_DIR / f"{task}_{kind}.yaml", "w") as f:
            f.write(f"# copied from {folder}/trainings/{run}/config.yaml\n")
            yaml.safe_dump(cfg, f, sort_keys=False)


def find_run(folder, prefix, before, seed, task):
    for d in sorted(set(os.listdir(ROOT / folder / "trainings")) - before):
        if not d.startswith(prefix):
            continue
        c = ROOT / folder / "trainings" / d / "config.yaml"
        if c.exists():
            cfg = yaml.safe_load(open(c))
            ds_task = (cfg.get("dataset") or {}).get("task", "vki")
            if cfg.get("seed") == seed and (task == "vki" or ds_task == task):
                return d
    raise RuntimeError(f"no new {prefix} folder with seed={seed} task={task}")


def run(task, seed):
    folder = REF[task][0]
    env = dict(os.environ, WANDB_MODE="disabled", HYDRA_FULL_ERROR="1")
    cwd = ROOT / folder
    extra = ["optim.epochs=1"] if os.environ.get("MARIO_SMOKE") else []   # 1-epoch smoke test
    rec = dict(task=task, seed=seed, host=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    before = set(os.listdir(cwd / "trainings"))
    t0 = time.time()
    subprocess.run([PY, "train_sdf.py", "--config-path", str(CFG_DIR), "--config-name", f"{task}_sdf", f"+seed={seed}"] + extra,
                   cwd=cwd, env=env, check=True)
    rec["sdf_seconds"] = round(time.time() - t0)
    sdf_dir = find_run(folder, "training_sdf_", before, seed, task)
    mods = cwd / "trainings" / sdf_dir / "modulations"
    tr, te = LATENT[task]
    before = set(os.listdir(cwd / "trainings"))
    t0 = time.time()
    subprocess.run([PY, "train.py", "--config-path", str(CFG_DIR), "--config-name", f"{task}_flow", f"+seed={seed}",
                    f"dataset.train_latents_path={mods / tr}", f"dataset.test_latents_path={mods / te}"] + extra,
                   cwd=cwd, env=env, check=True)
    rec["flow_seconds"] = round(time.time() - t0)
    rec["sdf_dir"] = f"{folder}/trainings/{sdf_dir}"
    rec["flow_dir"] = f"{folder}/trainings/{find_run(folder, 'training_result_', before, seed, task)}"
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec)); w.writeheader() if new else None; w.writerow(rec)
    print("DONE", rec, flush=True)


if __name__ == "__main__":
    jobs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
    for task in {t for t, _ in jobs}:
        prepare_configs(task)
    for task, seed in jobs:
        run(task, seed)
