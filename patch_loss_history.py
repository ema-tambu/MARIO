import re

path = "airfrans_task/train_sdf.py"
with open(path) as f:
    src = f.read()

src = src.replace(
    "    best_val = float('inf')\n",
    "    best_val = float('inf')\n"
    "    train_loss_history = []\n"
    "    val_loss_history = []\n"
    "    val_epochs = []\n"
)

src = src.replace(
    '        wandb.log({"epoch": epoch, "train_loss": avg_train})\n',
    '        wandb.log({"epoch": epoch, "train_loss": avg_train})\n'
    '        train_loss_history.append(avg_train)\n'
)

src = src.replace(
    '            wandb.log({"epoch": epoch, "val_loss": avg_val})\n',
    '            wandb.log({"epoch": epoch, "val_loss": avg_val})\n'
    '            val_loss_history.append(avg_val)\n'
    '            val_epochs.append(epoch)\n'
)

src = src.replace(
    '    print(f"Saved modulations: train {train_mods.shape}, test {test_mods.shape}")\n',
    '    print(f"Saved modulations: train {train_mods.shape}, test {test_mods.shape}")\n'
    '    np.savez(results_dir / "loss_history.npz",\n'
    '             train_loss=np.array(train_loss_history),\n'
    '             val_loss=np.array(val_loss_history),\n'
    '             val_epochs=np.array(val_epochs))\n'
    '    print(f"Saved loss history: {len(train_loss_history)} train pts, {len(val_loss_history)} val pts")\n'
)

with open(path, "w") as f:
    f.write(src)
print("Patched successfully.")
