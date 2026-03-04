import matplotlib.pyplot as plt
import os
import lasp.metrics
import numpy as np
import src.utils.Utils as Utils

def plot_metrics(metrics, output_dir):
    for key, values in metrics.items():
        plt.figure()
        if isinstance(values, dict):
            for subkey, series in values.items():
                series = np.asarray(series)
                if series.ndim != 1:
                    # aplatir si besoin
                    series = series.ravel()
                plt.plot(range(1, len(series) + 1), series, linestyle='-', label=f"block_{subkey}")

            plt.title(f"Évolution de {key} (par block)")
            plt.xlabel("Époque")
            plt.ylabel(key)
            plt.legend(title="Block", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            series = np.asarray(values)
            if series.size == 0:
                print(f"Avertissement: la métrique {key} est vide, skip.")
                plt.close()
                continue
            # Si ce n'est pas 1D on essaye d'aplatir proprement
            if series.ndim != 1:
                series = series.ravel()
            plt.plot(range(1, len(series) + 1), series, linestyle='-')
            plt.title(f"Évolution de {key}")
            plt.xlabel("Époque")
            plt.ylabel(key)
            plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{key}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Figure sauvegardée : {save_path}")

def compute_metrics(original, output):
    res_size = original.shape
    inp_size = output.shape
    img_low_res = output
    if res_size != inp_size :
        decim_row = res_size[0] // inp_size[0]
        decim_col = res_size[1] // inp_size[1]
        nb_row, nb_col = output.shape
        img_low_res = np.zeros((decim_row * nb_row, decim_col * nb_col), dtype=output.dtype)
        img_low_res[::decim_row, ::decim_col] = output.copy()
    psnr = lasp.metrics.PSNR(original, img_low_res, intensity_max=1)
    mse = lasp.metrics.MSE(original, img_low_res)
    mae = lasp.metrics.MAE(original, img_low_res)
    maxae = np.max(np.abs(original - img_low_res))
    return psnr, mse, mae, maxae

def show_and_save_3images(original, input_normalized, output, output_dir, id_img, params):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Calcul des métriques principales ---
    psnr_input, mse_input, mae_input, maxae_input = compute_metrics(original, input_normalized)
    psnr_output, mse_output, mae_output, maxae_output = compute_metrics(original, output)

    # --- Affichage texte ---
    psnr_input_str = f"{psnr_input:.2f}"
    mse_input_str  = f"{mse_input:.4f}"
    mae_input_str  = f"{mae_input:.4f}"
    maxae_input_str = f"{maxae_input:.2f}"

    psnr_output_str = f"{psnr_output:.2f}"
    mse_output_str  = f"{mse_output:.4f}"
    mae_output_str  = f"{mae_output:.4f}"
    maxae_output_str = f"{maxae_output:.2f}"

    title_full = (
        f"Mumford-Shah\n"
        f"Blur filter: {params[0]}x{params[0]}, σ={params[1]} | Decimation: {params[2]}x{params[2]} | SNRdB: {params[4]}\n"
        f"Input (Low-Res) : PSNR: {psnr_input_str} | MSE: {mse_input_str} | MAE: {mae_input_str} | MaxAE: {maxae_input_str}\n"
        f"Reconstruction (High-Res) : PSNR: {psnr_output_str} | MSE: {mse_output_str} | MAE: {mae_output_str} | MaxAE: {maxae_output_str}\n"
    )
    fig.suptitle(title_full, fontsize=16, y=1.02)

    # --- Affichage des 3 images ---
    axes = fig.subplots(1, 3)
    images = [
        (axes[0], original, "Original"),
        (axes[1], input_normalized, "Input (Low-Res)"),
        (axes[2], output, "Reconstruction (High-Res)")
    ]

    for ax, img, title in images:
        #ax.axis("off")
        ax.set_title(title, fontsize=14)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])


    # --- Sauvegarde ---
    save_path = os.path.join(output_dir, f"{id_img}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    plot_histogram_gray(output, os.path.join(output_dir, f"{id_img}_hist.png"))
    return (
        psnr_input, mse_input, mae_input, maxae_input,
        psnr_output, mse_output, mae_output, maxae_output
    )

def plot_histogram_gray(image, filename):
    """
    Affiche et sauvegarde l'histogramme d'une image en niveaux de gris (valeurs entre 0 et 1).

    Parameters:
        image (np.ndarray) : tableau 2D ou 3D d'image avec valeurs entre 0 et 1.
        filename (str) : nom du fichier de sortie (ex: "hist.png").
    """

    # Vérification que les valeurs sont bien dans [0,1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("L'image doit contenir des valeurs entre 0 et 1.")

    plt.figure(figsize=(6,4))
    plt.hist(image.flatten(), bins=50, range=(0,1))
    plt.title("Histogramme des niveaux de gris")
    plt.xlabel("Valeur (0 = noir, 1 = blanc)")
    plt.ylabel("Nombre de pixels")

    # Sauvegarde dans un fichier
    plt.savefig(filename, dpi=300)
    plt.close