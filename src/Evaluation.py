from src.utils.UtilsPlot import show_and_save_3images
import torch
import numpy as np
import pandas as pd
import os

def evaluation(model, evaluation_loader, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Stockage des métriques
    psnr_input_list, mse_input_list, mae_input_list, maxae_input_list = [], [], [], []
    psnr_output_list, mse_output_list, mae_output_list, maxae_output_list = [], [], [], []

    with torch.no_grad():
        for _, (O, L, P) in enumerate(evaluation_loader):
            for i in range(len(O)):
                original_true  = O[i].to(device)
                low_resolution = L[i].to(device)
                params = P[i].to(device)

                res_size = original_true.size()
                inp_size = low_resolution.size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]
                original_pred = model(low_resolution, decim_row, decim_col)

                (
                    psnr_in, mse_in, mae_in, maxae_in,
                    psnr_out, mse_out, mae_out, maxae_out
                ) = show_and_save_3images(
                    original_true.cpu().numpy(),
                    low_resolution.cpu().numpy(),
                    original_pred.cpu().numpy(),
                    output_dir,
                    len(psnr_output_list),
                    params
                )

                psnr_input_list.append(psnr_in)
                mse_input_list.append(mse_in)
                mae_input_list.append(mae_in)
                maxae_input_list.append(maxae_in)

                psnr_output_list.append(psnr_out)
                mse_output_list.append(mse_out)
                mae_output_list.append(mae_out)
                maxae_output_list.append(maxae_out)

    df = pd.DataFrame({
        "Image_ID": list(range(len(psnr_output_list))),
        "PSNR_Input": psnr_input_list,
        "PSNR_Output": psnr_output_list,
        "MSE_Input": mse_input_list,
        "MSE_Output": mse_output_list,
        "MAE_Input": mae_input_list,
        "MAE_Output": mae_output_list,
        "MaxAE_Input": maxae_input_list,
        "MaxAE_Output": maxae_output_list,
    })

    df.loc[len(df)] = (
        "Moyenne",
        np.mean(psnr_input_list), np.mean(psnr_output_list),
        np.mean(mse_input_list), np.mean(mse_output_list),
        np.mean(mae_input_list), np.mean(mae_output_list),
        np.mean(maxae_input_list), np.mean(maxae_output_list)
    )

    df = df.round(3)

    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n📊 Fichier de métriques sauvegardé : {csv_path}")
