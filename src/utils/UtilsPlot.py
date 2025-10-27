import matplotlib.pyplot as plt
import os

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