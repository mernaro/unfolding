import torch

def train_epoch(model, optimizer, criterion, train_loader, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_train_loss = 0.0
    model.train()
    
    for _, (O, L) in enumerate(train_loader):
        optimizer.zero_grad()
        for i in range(batch_size):
            original_true = O[i].to(device)
            low_resolution = L[i].to(device)

            res_size = original_true[i].size()
            inp_size = low_resolution[i].size()
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
            for i in range(batch_size):
                original_true = O[i].to(device)
                low_resolution = L[i].to(device)
    
                res_size = original_true[i].size()
                inp_size = low_resolution[i].size()
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

def train(model, optimizer, criterion, train_loader, batch_size_train, validation_loader, batch_size_validation, nb_epoch, patience):
    best_validation_loss = float("inf")
    epoch_no_improve = 0
    for epoch in range(nb_epoch):
        avg_train_loss = train_epoch(model, oprimizer, criterion, train_loader, batch_size_train)
        avg_validation_loss = validation_epoch(model, optimizer, criterion, validation_loader, batch_size_validation)
        best_validation_loss, epoch_no_improve = early_stop(best_validation_loss, avg_validation_loss, epoch_no_improve)
        if epoch_no_improve == patience :
            break