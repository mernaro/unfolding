from models.Unfolding import Unfolding
from src.Train import train
from src.datasets.ImageDataset import ImageDataset
from src.utils.UtilsLauncher import json_reader, data_config_reader,add_dated_folder, json_saver
from torch.utils import data
from torch.optim import Adam
import torch.nn
import argparse

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
    data_dir, train_instances, validation_instances, evaluation_instances = data_config_reader(config)
    output_dir = add_dated_folder(config["output_dir"])

    print("Initialisation des datasets...")
    train_dataset = ImageDataset(train_instances,'train',data_dir=data_dir)
    train_loader = data.DataLoader(train_dataset,batch_size=train_config["training_batch_size"],shuffle=True)

    val_dataset = ImageDataset(validation_instances,'validation',data_dir=data_dir)
    val_loader = data.DataLoader(val_dataset,batch_size=train_config["validation_batch_size"],shuffle=True)

    evaluation_dataset = ImageDataset(evaluation_instances,'validation',data_dir=data_dir)
    evaluation_loader = data.DataLoader(evaluation_dataset,batch_size=train_config["validation_batch_size"],shuffle=True)
    print(f"Datasets initialisés : train({len(train_dataset)}), validation ({len(val_dataset)}), evaluation({len(evaluation_dataset)}).")

    print("Initialisation du modèle...")
    model = Unfolding.from_config(config)
    print("Modèle initialisé.")

    print("Initialisation de l’optimiseur et de la fonction de perte...")
    optimizer = Adam(model.parameters(), lr=train_config["learning_rate"])
    criterion = torch.nn.MSELoss()
    print("Optimiseur et fonciton de perte initialisés.")

    nb_epochs = train_config["nb_epochs"]
    print(f"Début de l'entraînement pour {nb_epochs} époques...")
    train(model,optimizer,criterion,train_loader,32,val_loader,32,nb_epochs,2,output_dir)
    print("=== Entraînement terminé avec succès ===")
    json_saver(output_dir, config)