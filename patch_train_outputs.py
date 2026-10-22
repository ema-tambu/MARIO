path = "airfrans_task/train.py"
with open(path) as f:
    src = f.read()

old_save_block = '''    save_path = os.path.join('/scratch/dmsm/gi.catalani/Airfrans/', f"{run_name}_predictions.pt")
    torch.save(results, save_path)
    print(f"Saved predictions to {save_path}")
    return'''

new_save_block = '''    import pickle
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
    return'''

assert old_save_block in src, "pattern not found!"
src = src.replace(old_save_block, new_save_block)

with open(path, "w") as f:
    f.write(src)
print("Patched train.py successfully.")
