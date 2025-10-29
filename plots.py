import json
import os
import matplotlib.pyplot as plt

def plot_training_history(json_path, save_path=None, show=True):
    """
    Plot training and validation loss & accuracy curves from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing training metrics.
        save_path (str, optional): Path to save the resulting plot image. If None, it won't be saved.
        show (bool): Whether to display the plot after creation.

    Example:
        plot_training_history("metrics/pathvqa_yesno_model_s10.json")
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file: {json_path}")

    # Load the metrics JSON
    with open(json_path, "r") as f:
        metrics = json.load(f)

    epochs = metrics["epoch"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]

    # Create figure
    plt.figure(figsize=(10, 8))

    # --- Loss Plot ---
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Accuracy Plot ---
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example 2: Save the plot to a file instead of showing it
    plot_training_history(
        "metrics/pathvqa_yesno_model_s10.json",
        save_path="plots/training_curves_s10.png",
        show=False
    )
