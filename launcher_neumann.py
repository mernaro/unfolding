from models.NeumannNet import NeumannNet
from src.Train import train
from src.Evaluation import evaluation
from src.datasets.ImageDataset import ImageDataset, get_batch_with_variable_size_image
from src.utils.UtilsLauncher import json_reader, data_config_reader, add_dated_folder, json_saver
from torch.utils import data
from torch.optim import Adam, AdamW
import torch.nn
import argparse
import os

if __name__ == '__main__':
    print("=== Lancement du script principal (NeumannNet) ===")

    parser = argparse.ArgumentParser(
        description='Training a model to super resolve an image'
    )
    parser.add_argument("-c", "--config",
                        help="Path to a specific config.",
                        default="/projects/memaro/mcodjo/unfolding/config_neumann.json")
    parser.add_argument("-a", "--action",
                        help="The action of the program. Default: train",
                        default="train")
    args = parser.parse_args()

    print('\n-- ARGUMENT DU PROGRAMME')
    print(f'\t| Config : {args.config}')
    print(f'\t| Action : {args.action}')

    config       = json_reader(args.config)
    train_config = config["train"]
    output_dir   = add_dated_folder(config["output_dir"])

    data_dir, train_instances, validation_instances, evaluation_instances = data_config_reader(config)

    print("Initialisation des datasets...")
    train_dataset      = ImageDataset(train_instances,      "train", data_dir=data_dir)
    val_dataset        = ImageDataset(validation_instances, "val",   data_dir=data_dir)
    evaluation_dataset = ImageDataset(evaluation_instances, "test",  data_dir=data_dir)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size = train_config["training_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size = train_config["validation_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = True,
    )
    evaluation_loader = data.DataLoader(
        evaluation_dataset,
        batch_size = train_config["validation_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = False,
    )
    print(f"Datasets initialises : train({len(train_dataset)}), "
          f"validation({len(val_dataset)}), evaluation({len(evaluation_dataset)}).")

    if args.action == "train":
        print("Initialisation du modele NeumannNet...")
        model = NeumannNet.from_config(config)
        print("Modele initialise.")

        print("Initialisation de l optimiseur et de la fonction de perte...")
        if "weight_decay" not in train_config:
            optimizer = Adam(model.parameters(), lr=train_config["learning_rate"])
        else:
            optimizer = AdamW(model.parameters(),
                              lr=train_config["learning_rate"],
                              weight_decay=train_config["weight_decay"])
            print("Optimiseur avec weight decay.")
        criterion = torch.nn.MSELoss()
        print("Optimiseur et fonction de perte initialises.")

        nb_epochs = train_config["nb_epochs"]
        patience  = train_config["patience"]
        min_delta = train_config["min_delta"]

        print(f"Debut de l entrainement pour {nb_epochs} epoques...")
        train(model, optimizer, criterion,
              train_loader, train_config["training_batch_size"],
              val_loader,   train_config["validation_batch_size"],
              nb_epochs, patience, output_dir, min_delta)

        json_saver(output_dir, config)
        print("=== Entrainement termine avec succes ===")

    elif args.action == "test":
        print("Initialisation du modele NeumannNet...")
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        model      = NeumannNet.from_config(config)
        state_dict = torch.load(config["model_dir"], map_location=device)
        model.load_state_dict(state_dict)
        print("Modele initialise.")

        print("Debut de l evaluation du modele...")
        evaluation(model, evaluation_loader, output_dir)
        print("=== Evaluation terminee avec succes ===")