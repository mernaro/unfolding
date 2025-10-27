import matplotlib.pyplot as plt
import os
import lasp.metrics
import numpy as np

def plot_metrics(metrics, output_dir):
    for key, values in metrics.items():
        plt.figure()
        plt.plot(range(1, len(values) + 1), values, marker='o')
        plt.title(f"Évolution de {key}")
        plt.xlabel("Époque")
        plt.ylabel(key)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{key}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Figure sauvegardée : {save_path}")

def show_and_save_3images(original, input_normalized, output, output_dir, id_img):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    psrn = lasp.metrics.PSNR(original, output, intensity_max=255)
    mse = lasp.metrics.MSE(original, output)
    mae = lasp.metrics.MAE(original, output)
    
    psnr_str = f"{psrn:.2f}"
    mse_str  = f"{mse:.4f}"
    mae_str  = f"{mae:.4f}"

    title_full = f"Mumford-Shah\nPSNR: {psnr_str} | MSE: {mse_str} | MAE: {mae_str}"
    fig.suptitle(title_full, fontsize=16, y=1.02)
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

    save_path = os.path.join(output_dir, f"{id_img}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return psrn, mse, mae