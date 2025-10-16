import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from model_definition import PathVQAYesNoModel, PathVQAYesNoDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from logger import get_project_path

# Check if GPU is available
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def load_model(model_path):
    """Load trained model from path"""
    model = PathVQAYesNoModel()
    # Use weights_only=True to avoid the warning in future PyTorch versions
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(DEVICE),
                   weights_only=True))
    model = model.to(DEVICE)
    model.eval()  # Set model to evaluation mode
    return model


def evaluate_model(model_path, batch_size=32, logger=None):
    """Evaluate model on test dataset"""
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    model = load_model(model_path)
    test_dataset = PathVQAYesNoDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].float().unsqueeze(1).to(DEVICE)

            outputs = model(images, input_ids, attention_mask)
            preds = (outputs > 0.5).float()

            # Collect predictions and labels for metrics calculation
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total

    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Sensitivity is the same as recall (true positive rate)
    sensitivity = recall  # equals tp / (tp + fn)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }

    # Print all metrics
    log(f"Test Metrics:")
    log(f"  Accuracy: {accuracy:.4f}")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall: {recall:.4f}")
    log(f"  Sensitivity: {sensitivity:.4f}")
    log(f"  Specificity: {specificity:.4f}")
    log(f"  F1 Score: {f1:.4f}")
    log(f"  Confusion Matrix:")
    log(f"    True Negatives: {tn}")
    log(f"    False Positives: {fp}")
    log(f"    False Negatives: {fn}")
    log(f"    True Positives: {tp}")

    # Save metrics to file
    sample_count = extract_sample_count_from_path(model_path)
    sample_label = "All" if sample_count is None else sample_count
    metrics_path = os.path.join(get_project_path("metrics"),
                                f"test_metrics_s{sample_label}.json")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f"Test metrics saved to {metrics_path}")

    return metrics


def extract_sample_count_from_path(model_path):
    """Extract sample count from model path"""
    filename = os.path.basename(model_path)
    # Try to find s{number} pattern
    parts = filename.split('_')
    for part in parts:
        if part.startswith('s') and part[1:].isdigit():
            return part[1:]
    return None


def extract_batch_size_from_path(model_path):
    """Extract batch size from model path"""
    filename = os.path.basename(model_path)
    # Try to find b{number} pattern
    parts = filename.split('_')
    for part in parts:
        if part.startswith('b') and part[1:].isdigit():
            return part[1:]
    return "16"  # Default batch size if not found


def plot_metrics(model_path, test_metrics=None, save_path=None,
                 show_plot=True):
    """Plot all evaluation metrics in a comprehensive visualization"""
    sample_count = extract_sample_count_from_path(model_path)

    # If test metrics not provided, calculate them
    if test_metrics is None:
        print("Calculating test metrics...")
        test_metrics = evaluate_model(model_path)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Bar chart of performance metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall',
                     'Sensitivity', 'Specificity', 'F1 Score']
    metrics_values = [
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['sensitivity'],
        test_metrics['specificity'],
        test_metrics['f1']
    ]

    bars = ax1.bar(metrics_names, metrics_values,
                   color=['blue', 'green', 'red', 'magenta', 'purple',
                          'orange'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score')
    ax1.set_title(f'Performance Metrics (Sample Size: {sample_count})')
    ax1.grid(axis='y', alpha=0.3)

    # Second subplot: Confusion matrix as a heatmap
    cm = np.array([
        [test_metrics['confusion_matrix']['true_negatives'],
         test_metrics['confusion_matrix']['false_positives']],
        [test_metrics['confusion_matrix']['false_negatives'],
         test_metrics['confusion_matrix']['true_positives']]
    ])

    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Count')

    # Add text annotations to confusion matrix
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cm[i, j]}',
                     ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted No', 'Predicted Yes'])
    ax2.set_yticklabels(['Actual No', 'Actual Yes'])

    plt.tight_layout()

    # Save the plot if path is provided
    if save_path:
        plots_dir = os.path.dirname(save_path)
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    elif not show_plot:
        default_save_path = os.path.join(get_project_path("plots"),
                                         f"pathvqa_yesno_model_s{sample_count}_metrics.png")
        plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {default_save_path}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        # Close the plot to free resources
        plt.close()

    return test_metrics


def plot_accuracy_comparison(model_path, test_acc=None, save_path=None,
                             show_plot=True):
    """Plot training, validation, and test accuracy"""
    sample_count = extract_sample_count_from_path(model_path)
    batch_size = extract_batch_size_from_path(model_path)

    # Try multiple potential metrics file paths
    metrics_dir = get_project_path("metrics")
    possible_paths = [
        # New naming format (matches the model filename)
        os.path.join(metrics_dir,
                     os.path.basename(model_path).replace('.pt', '.json')),
        # Old naming format with s and b
        os.path.join(metrics_dir,
                     f"metrics_s{sample_count}_b{batch_size}.json"),
        # Simplified naming format with just s
        os.path.join(metrics_dir, f"pathvqa_yesno_model_s{sample_count}.json")
    ]

    metrics_path = None
    for path in possible_paths:
        if os.path.exists(path):
            metrics_path = path
            print(f"Found metrics at: {metrics_path}")
            break

    if not metrics_path:
        print(f"No metrics file found for {model_path}. Tried these paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    # If test metrics not provided, calculate them
    if test_acc is None:
        print("Calculating test metrics...")
        test_metrics = evaluate_model(model_path)
        test_acc = test_metrics['accuracy']
    elif isinstance(test_acc, dict):
        # If test_acc is actually a metrics dictionary
        test_acc = test_acc['accuracy']

    try:
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['train_acc'], 'b-', linewidth=2,
                 label='Training Accuracy')
        plt.plot(metrics['epoch'], metrics['val_acc'], 'r-', linewidth=2,
                 label='Validation Accuracy')

        # Show final validation accuracy
        final_val_acc = metrics['val_acc'][-1]
        plt.annotate(f"Val: {final_val_acc:.4f}",
                     xy=(metrics['epoch'][-1], final_val_acc),
                     xytext=(5, 5), textcoords='offset points')

        # Add a horizontal line for test accuracy
        plt.axhline(y=test_acc, color='g', linestyle='--', linewidth=2,
                    label=f'Test Accuracy: {test_acc:.4f}')

        # Set plot formatting
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Model Accuracy (Sample Size: {sample_count})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot if path is provided
        if save_path:
            plots_dir = os.path.dirname(save_path)
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        elif not show_plot:
            default_save_path = os.path.join(get_project_path("plots"),
                                             f"pathvqa_yesno_model_s{sample_count}_accuracy.png")
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {default_save_path}")

        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            # Close the plot to free resources
            plt.close()

        # Check model generalization
        acc_diff = abs(final_val_acc - test_acc)
        if final_val_acc - test_acc > 0.05:
            print(
                f"Potential overfitting detected: Val acc ({final_val_acc:.4f}) is significantly higher than test acc ({test_acc:.4f})")
        elif test_acc - final_val_acc > 0.05:
            print(
                f"Unusual pattern: Test acc ({test_acc:.4f}) is significantly higher than val acc ({final_val_acc:.4f})")
        else:
            print(
                f"Model generalizes well: Val acc ({final_val_acc:.4f}) is close to test acc ({test_acc:.4f})")

    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_path}")
    except Exception as e:
        print(f"Error plotting accuracy: {e}")


if __name__ == "__main__":
    # Example of standalone usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate PathVQA Yes/No model and plot results")
    parser.add_argument("--model",
                        help="Path to the model file")
    parser.add_argument("--save", default=None,
                        help="Path to save the plot (optional)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display the plot (useful for scripts)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only calculate metrics without plotting accuracy curves")

    args = parser.parse_args()

    # Run evaluation and plotting
    test_metrics = evaluate_model(args.model)

    if not args.metrics_only:
        # Plot the accuracy curves
        plot_accuracy_comparison(args.model,
                                 test_acc=test_metrics,
                                 save_path=args.save,
                                 show_plot=not args.no_show)

    # Plot metrics visualization
    metrics_plot_path = None
    if args.save:
        metrics_plot_path = args.save.replace('.png', '_metrics.png')

    plot_metrics(args.model,
                 test_metrics=test_metrics,
                 save_path=metrics_plot_path,
                 show_plot=not args.no_show)
