# ---- patch train_sdf.py ----
path = "train_sdf.py"
with open(path) as f:
    src = f.read()

src = src.replace(
    "import airfrans as af\nfrom dataset import SDFDataset, subsample_dataset",
    "from dataset import SDFDataset, subsample_dataset, load_vkils59"
)

src = src.replace(
    "    train_data, train_names = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=True)\n"
    "    test_data, test_names = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=False)",
    "    train_data, train_names = load_vkils59(cfg.dataset.root_path, train=True)\n"
    "    test_data, test_names = load_vkils59(cfg.dataset.root_path, train=False)"
)

src = src.replace(
    'np.savez(mod_dir / f"{cfg.dataset.task}_train_{cfg.inr.latent_dim}.npz",',
    'np.savez(mod_dir / f"vkils59_train_{cfg.inr.latent_dim}.npz",'
)
src = src.replace(
    'np.savez(mod_dir / f"{cfg.dataset.task}_test_{cfg.inr.latent_dim}.npz",',
    'np.savez(mod_dir / f"vkils59_test_{cfg.inr.latent_dim}.npz",'
)

with open(path, "w") as f:
    f.write(src)
print("Patched train_sdf.py")

# ---- patch train.py ----
path = "train.py"
with open(path) as f:
    src = f.read()

src = src.replace(
    "import airfrans as af",
    ""
)
src = src.replace(
    "from dataset import AirfransFlowDataset, subsample_dataset",
    "from dataset import VKILS59FlowDataset, subsample_dataset, load_vkils59"
)

src = src.replace(
    "    train_data, train_names = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=True)\n"
    "    test_data, test_names = af.dataset.load(root=cfg.dataset.root_path, task=cfg.dataset.task, train=False)  ",
    "    train_data, train_names = load_vkils59(cfg.dataset.root_path, train=True)\n"
    "    test_data, test_names = load_vkils59(cfg.dataset.root_path, train=False)"
)

src = src.replace("AirfransFlowDataset(", "VKILS59FlowDataset(")

src = src.replace(
    "    output_fields = ['Velocity-x', 'Velocity-y', 'Pressure', 'Turbulent-viscosity']  # Same order as in the dataset class",
    "    output_fields = ['Mach', 'Turbulent-viscosity']  # Same order as in the dataset class"
)

with open(path, "w") as f:
    f.write(src)
print("Patched train.py")
