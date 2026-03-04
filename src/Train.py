import torch
import os
from src.utils.UtilsPlot import plot_metrics
import time


def train_epoch(model, optimizer, criterion, train_loader, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_train_loss = 0.0
    nb_ite = 0
    model.train()

    for _, (O, L, params_list) in enumerate(train_loader):
        optimizer.zero_grad()
        nb_ite += len(O)

        for i in range(len(O)):
            original_true = O[i].to(device)   # [C, H_hr, W_hr]
            low_res       = L[i].to(device)   # [C, H_lr, W_lr]
            params        = params_list[i]    # tensor([blur_size, blur_sigma, decimation, noise_value, noise_db])

            # Normalisation de forme : [H, W] → [1, H, W]
            if original_true.dim() == 2:
                original_true = original_true.unsqueeze(0)
            if low_res.dim() == 2:
                low_res = low_res.unsqueeze(0)

            # params[2] = decimation
            decim = int(params[2].item())
            res_size = original_true.size()   # [C, H_hr, W_hr]
            inp_size = low_res.size()         # [C, H_lr, W_lr]
            decim_row = res_size[-2] // inp_size[-2]
            decim_col = res_size[-1] // inp_size[-1]

            # Niveau de bruit 
            sigma = params[3].item()   # noise_value

            original_pred = model(low_res, decim_row, decim_col, sigma)   # [C, H_hr, W_hr]

            loss = criterion(original_pred, original_true)
            avg_train_loss += loss.item()
            loss.backward()

        optimizer.step()

    avg_train_loss /= nb_ite
    return avg_train_loss


def validation_epoch(model, optimizer, criterion, validation_loader, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_validation_loss = 0.0
    nb_ite = 0
    model.eval()

    with torch.no_grad():
        for _, (O, L, params_list) in enumerate(validation_loader):
            nb_ite += len(O)

            for i in range(len(O)):
                original_true = O[i].to(device)
                low_res       = L[i].to(device)
                params        = params_list[i]

                # Normalisation de forme : [H, W] → [1, H, W]
                if original_true.dim() == 2:
                    original_true = original_true.unsqueeze(0)
                if low_res.dim() == 2:
                    low_res = low_res.unsqueeze(0)

                res_size  = original_true.size()
                inp_size  = low_res.size()
                decim_row = res_size[-2] // inp_size[-2]
                decim_col = res_size[-1] // inp_size[-1]
                sigma     = params[3].item()

                original_pred = model(low_res, decim_row, decim_col, sigma)

                loss = criterion(original_pred, original_true)
                avg_validation_loss += loss.item()

    avg_validation_loss /= nb_ite
    return avg_validation_loss


def early_stop(best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta):
    if avg_validation_loss + min_delta < best_validation_loss:
        best_validation_loss = avg_validation_loss
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
    return best_validation_loss, epoch_no_improve


def train(model, optimizer, criterion, train_loader, batch_size_train,
          validation_loader, batch_size_validation, nb_epoch, patience, output_dir, min_delta):

    start_time_total = time.time()
    best_validation_loss = float("inf")
    epoch_no_improve = 0
    best_model_state = None
    epoch_save = 0
    metrics = {"train_loss": [], "validation_loss": []}

    for epoch in range(nb_epoch):
        print(f"\n--- Epoque {epoch + 1}/{nb_epoch} ---")

        avg_train_loss      = train_epoch(model, optimizer, criterion, train_loader, batch_size_train)
        avg_validation_loss = validation_epoch(model, optimizer, criterion, validation_loader, batch_size_validation)

        metrics["train_loss"].append(avg_train_loss)
        metrics["validation_loss"].append(avg_validation_loss)
        metrics.update(model.get_metrics())  

        print(f"  Train loss      : {avg_train_loss:.6f}")
        print(f"  Validation loss : {avg_validation_loss:.6f}")
        print(f"  η (eta)         : {model.eta.item():.6f}")

        best_validation_loss, epoch_no_improve = early_stop(
            best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta
        )

        if epoch_no_improve == 0:
            best_model_state = model.state_dict()
            epoch_save = epoch + 1

        if epoch_no_improve == patience:
            print(f"[EARLY STOP] Arret à l'epoque {epoch + 1} — pas d'amelioration depuis {patience} epoques.")
            break

    total_duration = time.time() - start_time_total
    torch.save(best_model_state, os.path.join(output_dir, "best_model.pth"))
    print(f"\n[FIN] Entrainement termine en {total_duration:.2f}s "
          f"({total_duration/60:.2f} min, {total_duration/3600:.2f} h).")
    print(f"[SAVE] Meilleur modele sauvegarde dans {os.path.join(output_dir, 'best_model.pth')} "
          f"à l'époque n°{epoch_save}.")
    plot_metrics(metrics, output_dir)