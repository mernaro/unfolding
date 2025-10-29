from models.Unfolding import Unfolding
from src.Train import train
from src.Evaluation import evaluation
from src.datasets.ImageDataset import ImageDataset, get_batch_with_variable_size_image
from src.utils.UtilsLauncher import json_reader, data_config_reader,add_dated_folder, json_saver
from torch.utils import data
from torch.optim import Adam
import torch.nn
import argparse
import os

if __name__ == '__main__' :
    print("=== Lancement du script principal ===")
    parser = argparse.ArgumentParser(
		description='Training a model to super resolve an image'
	)

    parser.add_argument("-c", "--config",
                        help="Path to a specific config. Default: /projects/memaro/rpujol/unfolding/config.json",
                        default="/projects/memaro/rpujol/unfolding/config.json")

    parser.add_argument("-a", "--action",
                        help="The action of the program. Default: train",
                        default="train")

    args = parser.parse_args()
    config_path = args.config
    action = args.action
    print('\n-- ARGUMENT DU PROGRAME')
    print(f'\t| Config : {config_path}')
    print(f'\t| Action : {action}')
    
    config = json_reader(config_path)
    data_config = config["data"]
    train_config = config["train"]
    output_dir = add_dated_folder(config["output_dir"])
    data_dir, train_instances, validation_instances, evaluation_instances = data_config_reader(config)

    print("Initialisation des datasets...")
    dataset = ImageDataset(train_instances,'train',data_dir=data_dir)
    train_size = int(0.7 * len(dataset)) 
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, evaluation_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    
    train_loader = data.DataLoader(train_dataset,batch_size=train_config["training_batch_size"],collate_fn = get_batch_with_variable_size_image,shuffle=True)
    val_loader = data.DataLoader(val_dataset,batch_size=train_config["validation_batch_size"],collate_fn = get_batch_with_variable_size_image,shuffle=True)
    evaluation_loader = data.DataLoader(evaluation_dataset,batch_size=train_config["validation_batch_size"],collate_fn = get_batch_with_variable_size_image,shuffle=True)
    print(f"Datasets initialisés : train({len(train_dataset)}), validation ({len(val_dataset)}), evaluation({len(evaluation_dataset)}).")

    if action == "train" :
        print("Initialisation du modèle...")
        model = Unfolding.from_config(config)
        print("Modèle initialisé.")
    
        print("Initialisation de l’optimiseur et de la fonction de perte...")
        optimizer = Adam(model.parameters(), lr=train_config["learning_rate"])
        criterion = torch.nn.MSELoss()
        print("Optimiseur et fonciton de perte initialisés.")
    
        nb_epochs = train_config["nb_epochs"]
        patience = train_config["patience"]
        print(f"Début de l'entraînement pour {nb_epochs} époques...")
        train(model,optimizer,criterion,train_loader,32,val_loader,32,nb_epochs,patience,output_dir)
        json_saver(output_dir, config)
        print("=== Entraînement terminé avec succès ===")
        
    elif action == "test" :
        print("Initialisation du modèle...")
        model_dir = config["model_dir"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Unfolding.from_config(config)
        state_dict = torch.load(model_dir, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("Modèle initialisé.")

        print(f"Début de l'évaluation du modèle...")
        evaluation(model,evaluation_loader,output_dir)
        print("=== Evaluation terminé avec succès ===")