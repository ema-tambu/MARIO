import os
import matplotlib.pyplot as plt
import sys
import yaml
from pathlib import Path
from pickletools import OpcodeInfo
import time
sys.path.append(str(Path(__file__).parents[1]))
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from src.load_models import create_inr_instance,load_inr
from torch_geometric.loader import DataLoader
import wandb
import pyoche as pch
from dataset import VKILS59FlowDataset, subsample_dataset, load_vkils59
from src.utils_training import training_step
from torch.utils.data import Subset



@hydra.main(config_path="", config_name="config_out.yaml")
def main(cfg: DictConfig) -> None:
    # optional seed (TMLR revision): pass +seed=<int> on the command line
    _seed = cfg.get("seed", None)
    if _seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(_seed); _np.random.seed(_seed)
        torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
        print(f"Seed set to {_seed}")
    # Initialize wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project='vkils59_flow', config=OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update({
        "batch_size": cfg.optim.batch_size,
        "lr_inr": cfg.optim.lr_inr,
        "epochs": cfg.optim.epochs,
        "num_points": cfg.optim.num_points,
        "latent_dim": cfg.inr.latent_dim,
        "model_type": cfg.inr.model_type
    })

    torch.set_default_dtype(torch.float32)

    # optim
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr 
    epochs = cfg.optim.epochs
    num_points = cfg.optim.num_points
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inr
    latent_dim = cfg.inr.latent_dim
    in_dim = cfg.inr.in_dim
    out_dim = cfg.inr.out_dim
    latent_cond = True
    predict_scalars = True if cfg.inr.out_scalar_dim > 0 else False
    
    
    train_data, train_names = load_vkils59(cfg.dataset.root_path, train=True)
    test_data, test_names = load_vkils59(cfg.dataset.root_path, train=False)
    train_latents = np.load(cfg.dataset.train_latents_path)
    test_latents = np.load(cfg.dataset.test_latents_path)
    # Create dataset instance
    train_dataset = VKILS59FlowDataset(
        train_data,
        train_names,
        geom_latents = train_latents['train_modulations'],
        mode         = 'train',
    )
    
    coef_norm = train_dataset.coef_norm

    # 5) Build the TEST-mode dataset (reusing the same stats from full_train_ds)
    test_dataset = VKILS59FlowDataset(
        test_data,
        test_names,
        geom_latents = test_latents['val_modulations'],
        mode         = 'test',
        coef_norm    = coef_norm)
    
    #Optionally validate directly on test set
    val_dataset = test_dataset
    
    ntrain = len(train_dataset)
    nval = len(val_dataset)
    ntest = len(test_dataset)

    inr_in = create_inr_instance(cfg, input_dim=in_dim, output_dim=out_dim, device=device)
    
    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
        ],
        lr=lr_inr,
        weight_decay=0,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_in,
        mode='min',
        factor=0.8,      # Reduce LR when plateauing
        patience=10,     # Number of epochs to wait before reducing LR
        min_lr=1e-5    # Minimum LR threshold
    )
    
    
    # Improved folder naming with timestamp and all output target fields
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"training_result_{timestamp}" 
    dir = os.path.dirname(__file__)
    results_directory = os.path.join(dir + '/trainings', folder_name)
    os.makedirs(results_directory, exist_ok=True)
    cfg_save_path = os.path.join(results_directory, 'config.yaml')
    # Convert OmegaConf DictConfig to a native Python dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Set `resolve=True` to resolve all interpolations
    with open(cfg_save_path, 'w') as file:
        yaml.dump(cfg_dict, file, default_flow_style=False)

    # Generate run name
    run_name = f"{cfg.inr.model_type}_nl{cfg.inr.depth}_np{num_points}_epochs{epochs}_lr{lr_inr}_ldim{latent_dim}"
   
    best_loss = np.inf
    train_loss_history = []
    test_loss_history = []

    # Check if restart training
    if cfg.restart_training:
        if os.path.exists(cfg.saved_model_path):
            checkpoint = torch.load(cfg.saved_model_path, weights_only=False)
            inr_in.load_state_dict(checkpoint["inr_in"])
            optimizer_in.load_state_dict(checkpoint["optimizer_inr_in"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Restarting training from epoch {start_epoch}")
        else:
            print("Saved model not found. Starting training from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0
    
    start_time = time.time()
    for step in tqdm(range(start_epoch, epochs)):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        fit_field  = 0.0
        fit_test_field = 0.0
        comp_loss = np.zeros(2)  # 2 output fields: mach, nut
        test_loss_in  = 1
        # Initialize lists for loss history
        
        use_pred_loss = True
        
        # Prepare the training dataset for the new epoch
        # Subsample the training dataset
        train_subsample_dataset = subsample_dataset(train_dataset, num_points)
        train_loader = DataLoader(train_subsample_dataset, batch_size=batch_size, shuffle=True)
        
        print('Num points for training', len(train_subsample_dataset[0].input))
        for substep,graph in enumerate(train_loader): 
             
            n_samples = len(graph)
            inr_in.train()
            graph = graph.cuda() if torch.cuda.is_available() else graph
            
            outputs = training_step(
                inr_in,
                graph,
                latent_cond=latent_cond,
                predict_scalars=predict_scalars,
                return_reconstructions=False,
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            loss = outputs["loss"].cpu().detach()
            field_loss = outputs["field_loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples
            fit_field += field_loss.item() * n_samples

        
        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_field = fit_field / (ntrain)
        train_loss_history.append({'epoch': step, 'loss': train_loss_in, 'field_loss': train_loss_field})
        # Inside training loop
        wandb.log({"train_loss": train_loss_in, "epoch": step, "train_field_loss": train_loss_field})
        print('Train Loss', train_loss_in, 'Field Loss', train_loss_field)
       

        if use_pred_loss:
            # Prepare the training dataset for the new epoch
            val_subsample_dataset = subsample_dataset(val_dataset, num_points)
            val_loader = DataLoader(val_subsample_dataset, batch_size=batch_size, shuffle=True)
            for substep,graph in enumerate(val_loader):  
                n_samples = len(graph)
                inr_in.eval()
                graph = graph.cuda() if torch.cuda.is_available() else graph
                outputs = training_step(
                    inr_in,
                    graph,
                    latent_cond=latent_cond,
                    predict_scalars=predict_scalars,
                    return_reconstructions=False,
                )
                loss = outputs["loss"].cpu().detach()
                fit_test_mse_in += loss.item() * n_samples
                field_test_loss = outputs["field_loss"].cpu().detach()
                fit_train_mse_in += loss.item() * n_samples
                fit_test_field += field_test_loss.item() * n_samples
                comp_loss += outputs["field_loss_components"].cpu().numpy() * n_samples
            
            test_loss_in = fit_test_mse_in / (nval)
            test_loss_field = fit_test_field / (nval)
            test_comp_loss = comp_loss / (nval)

            
            scheduler.step(test_loss_in)
            
            test_loss_history.append({'epoch': step, 'loss': test_loss_in, 'field_loss': test_loss_field})
            wandb.log({"val_loss": test_loss_in, 
                        "epoch": step, 
                        "val_field_loss": test_loss_field,
                        "val_field_loss_mach": test_comp_loss[0],
                        "val_field_loss_nut": test_comp_loss[1]})
            print('Test Loss', test_loss_in, 'Field Loss', test_loss_field)

            plt.figure()
            plt.plot([p['epoch'] for p in train_loss_history], [p['loss'] for p in train_loss_history], label='Train Loss')
            plt.plot([p['epoch'] for p in test_loss_history], [p['loss'] for p in test_loss_history], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig(os.path.join(results_directory, f"{run_name}_loss_plot.png"))
            plt.close()
        
        if test_loss_in < best_loss:
            best_loss = test_loss_in
            
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "inr_in": inr_in.state_dict(),
                    "optimizer_inr_in": optimizer_in.state_dict(),
                },
                os.path.join(results_directory, f"{run_name}.pt"),
            )
             
    inr_model = load_inr(torch.load(os.path.join(results_directory, f"{run_name}.pt"), weights_only=False)["inr_in"], cfg, in_dim, out_dim, device=device)
    inr_model.eval()
        # —————————— TEST‐TIME INFERENCE & COLLECTION ——————————
    # Prepare containers
    all_field_preds   = []
    # Get normalization stats for inversion
    out_mean  = coef_norm['output']['mean']
    out_std   = coef_norm['output']['std']
    # Convert dictionaries to arrays ensuring the correct order of output fields
    output_fields = ['Mach', 'Turbulent-viscosity']  # Same order as in the dataset class
    out_mean_array = np.array([out_mean[field] for field in output_fields])
    out_std_array = np.array([out_std[field] for field in output_fields])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)
            # 1) Predict fields (and unnormalize)
            field_n = inr_model.modulated_forward(
                graph.input, 
                graph.cond[graph.batch]
            )                              # (N, 1), normalized
            field_n = field_n.cpu().numpy()  # Convert to NumPy array
            # invert normalization: (x * std + mean)
            field_unnorm = field_n * out_std_array + out_mean_array
            all_field_preds.append(field_unnorm)


    # —————————— SAVE EVERYTHING ——————————
    results = {
        # lists of NumPy arrays, one entry per test sample
        'fields_pred':  all_field_preds,  
    }

    import pickle
    fields_pred_obj = np.empty(len(all_field_preds), dtype=object)
    for i, arr in enumerate(all_field_preds):
        fields_pred_obj[i] = arr

    npz_path = os.path.join(results_directory, f"{run_name}_predictions.npz")
    np.savez(npz_path,
             fields_pred=fields_pred_obj,
             names=np.array(test_names),
             output_fields=np.array(output_fields))

    pkl_path = os.path.join(results_directory, f"{run_name}_predictions.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'fields_pred': all_field_preds,
            'names': test_names,
            'output_fields': output_fields,
            'coef_norm': coef_norm,
        }, f)

    loss_hist_path = os.path.join(results_directory, "flow_loss_history.npz")
    np.savez(loss_hist_path,
             train_epoch=np.array([p['epoch'] for p in train_loss_history]),
             train_loss=np.array([p['loss'] for p in train_loss_history]),
             train_field_loss=np.array([p['field_loss'] for p in train_loss_history]),
             test_epoch=np.array([p['epoch'] for p in test_loss_history]),
             test_loss=np.array([p['loss'] for p in test_loss_history]),
             test_field_loss=np.array([p['field_loss'] for p in test_loss_history]))

    print(f"Saved predictions to {npz_path} and {pkl_path}")
    print(f"Saved loss history to {loss_hist_path}")
    return


if __name__ == "__main__":
    main()