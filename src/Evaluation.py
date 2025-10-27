from src.utils.UtilsPlot import show_and_save_3images
import torch
import numpy as np

def evaluation(model,evaluation_loader,output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_validation_loss = 0.0
    model.eval()
    id_img = 0
    psnr_list = []
    mse_list =[]
    mae_list = []
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
    
                psnr, mse, mae = show_and_save_3images(original_true.cpu().numpy(), low_resolution.cpu().numpy(), original_pred.cpu().numpy(), output_dir, id_img)
                psnr_list.append(psnr)
                mse_list.append(mse)
                mae_list.append(mae)
                id_img += 1
    print(f"Les résultats moyens sont : \n\t-PSNR : {np.mean(psnr_list):.2f}\n\t-MSE : {np.mean(mse_list):.2f}\n\t-MAE : {np.mean(mae_list):.2f}")