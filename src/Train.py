import torch
import os
from src.utils.UtilsPlot import plot_metrics

def train_epoch(model, optimizer, criterion, train_loader, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_train_loss = 0.0
    model.train()
    
    for _, (O, L) in enumerate(train_loader):
        optimizer.zero_grad()
        for i in range(len(O)):
            original_true = O[i][0].to(device)
            low_resolution = L[i][0].to(device)
            
            res_size = original_true.size()
            inp_size = low_resolution.size()
            decim_row = res_size[0] // inp_size[0]
            decim_col = res_size[1] // inp_size[1]
            
            original_pred = model(low_resolution, decim_row, decim_col)

            loss = criterion(original_pred, original_true)
            avg_train_loss += loss.item()
            loss.backward()
        optimizer.step()
    
    avg_train_loss /= len(train_loader)
    return avg_train_loss

def validation_epoch(model, optimizer, criterion, validation_loader, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_validation_loss = 0.0
    model.eval()

    with torch.no_grad():
        for _, (O, L) in enumerate(validation_loader):
            for i in range(len(O)):
                original_true = O[i][0].to(device)
                low_resolution = L[i][0].to(device)
    
                res_size = original_true.size()
                inp_size = low_resolution.size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]
                original_pred = model(low_resolution, decim_row, decim_col)
    
                loss = criterion(original_pred, original_true)
                avg_validation_loss += loss.item()

    avg_validation_loss /= len(validation_loader)
    return avg_validation_loss

def early_stop(best_validation_loss, avg_validation_loss, epoch_no_improve):
    if avg_validation_loss < best_validation_loss :
        best_validation_loss = avg_validation_loss
        epoch_no_improve = 0
    else :
        epoch_no_improve += 1
    return best_validation_loss, epoch_no_improve

def train(model, optimizer, criterion, train_loader, batch_size_train, validation_loader, batch_size_validation, nb_epoch, patience, output_dir):
    best_validation_loss = float("inf")
    epoch_no_improve = 0
    best_model_state = None
    epoch_save = 0
    metrics = {"train_loss": [], "validation_loss": []}
    for epoch in range(nb_epoch):
        print(f"\n--- Époque {epoch + 1}/{nb_epoch} ---")
        avg_train_loss = train_epoch(model, optimizer, criterion, train_loader, batch_size_train)
        avg_validation_loss = validation_epoch(model, optimizer, criterion, validation_loader, batch_size_validation)

        metrics["train_loss"].append(avg_train_loss)
        metrics["validation_loss"].append(avg_validation_loss)
        metrics.update(model.get_metrics())
        
        best_validation_loss, epoch_no_improve = early_stop(best_validation_loss, avg_validation_loss, epoch_no_improve)
        if epoch_no_improve == 0 :
            best_model_state = model.state_dict()
            epoch_save = epoch+1
        if epoch_no_improve == patience :
            print(f"[EARLY STOP] Arrêt anticipé à l’époque {epoch + 1} car aucune amélioration depuis {patience} époques.")
            break
    torch.save(best_model_state, os.path.join(output_dir, "best_model.pth"))
    print(f"[SAVE] Meilleur modèle sauvegardé dans {os.path.join(output_dir, "best_model.pth")} à l'époque n°{epoch_save}.")
    plot_metrics(metrics, output_dir)