from src.utils.UtilsPlot import show_and_save_3images
import torch
import numpy as np
import pandas as pd
import os

def evaluation(model,evaluation_loader,output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_validation_loss = 0.0
    model.eval()
    psnr_list = []
    mse_list =[]
    mae_list = []
    maxae_list = []
    with torch.no_grad():
        for _, (O, L) in enumerate(evaluation_loader):
            for i in range(len(O)):
                original_true = O[i].to(device)
                low_resolution = L[i].to(device)
    
                res_size = original_true.size()
                inp_size = low_resolution.size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]
                original_pred = model(low_resolution, decim_row, decim_col)
    
                psnr, mse, mae, maxae = show_and_save_3images(original_true.cpu().numpy(),low_resolution.cpu().numpy(),original_pred.cpu().numpy(),output_dir,len(psnr_list))
                psnr_list.append(psnr)
                mse_list.append(mse)
                mae_list.append(mae)
                maxae_list.append(maxae)
    psnr_mean = np.mean(psnr_list)
    mse_mean = np.mean(mse_list)
    mae_mean = np.mean(mae_list)
    maxae_mean = np.mean(maxae_list)

    print(f"Les résultats moyens sont : "
          f"\n\t-PSNR  : {psnr_mean:.2f}"
          f"\n\t-MSE   : {mse_mean:.2f}"
          f"\n\t-MAE   : {mae_mean:.2f}"
          f"\n\t-MAXAE : {maxae_mean:.2f}")

    df = pd.DataFrame({
        "Image_ID": list(range(len(psnr_list))),
        "PSNR": psnr_list,
        "MSE": mse_list,
        "MAE": mae_list,
        "MaxAE": maxae_list
    })

    df.loc[len(df)] = ["Moyenne", psnr_mean, mse_mean, mae_mean, maxae_mean]
    df = df.round(2)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n📊 Fichier de métriques sauvegardé avec pandas : {csv_path}")