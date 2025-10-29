import matplotlib.pyplot as plt
import os
import lasp.metrics
import numpy as np

import os
import matplotlib.pyplot as plt
import numpy as np

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


def show_and_save_3images(original, input_normalized, output, output_dir, id_img):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Calcul des métriques principales ---
    psrn = lasp.metrics.PSNR(original, output, intensity_max=255)
    mse = lasp.metrics.MSE(original, output)
    mae = lasp.metrics.MAE(original, output)
    maxae = np.max(np.abs(original - output))

    # --- Affichage texte ---
    psnr_str = f"{psrn:.2f}"
    mse_str  = f"{mse:.4f}"
    mae_str  = f"{mae:.4f}"
    maxae_str = f"{maxae:.2f}"

    title_full = (
        f"Mumford-Shah\n"
        f"PSNR: {psnr_str} | MSE: {mse_str} | MAE: {mae_str} | MaxAE: {maxae_str}"
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
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        ax.imshow(img, cmap='gray')

    # --- Sauvegarde ---
    save_path = os.path.join(output_dir, f"{id_img}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return psrn, mse, mae, maxae