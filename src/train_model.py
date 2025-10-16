import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from model_definition import PathVQAYesNoModel, PathVQAYesNoDataset
from torch.utils.data import DataLoader
from logger import get_project_path, setup_logger

# Check if GPU is available
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def validate(model, dataloader, criterion):
    """
    Validate the model on the validation dataset

    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function

    Returns:
        val_loss, val_acc: Validation loss and accuracy
    """
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].float().unsqueeze(1).to(DEVICE)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

    val_loss = total_loss / len(dataloader)
    val_acc = correct / len(dataloader.dataset)
    return val_loss, val_acc


def train_with_params(num_samples, batch_size, num_epochs=3, lr=1e-4,
                      weight_decay=1e-5, dropout=0.3, save_model=True,
                      custom_model_path=None, logger=None,
                      use_combined_data=False, early_stopping=True,
                      patience=3):
    """
    Train a model with the given hyperparameters

    Args:
        num_samples: Number of samples to use for training
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs to train
        lr: Learning rate
        weight_decay: Weight decay for Adam optimizer
        dropout: Dropout rate for the model
        save_model: Whether to save the model
        custom_model_path: Custom path to save the model (if None, a default path is used)
        logger: Logger instance for recording output
        use_combined_data: If True, combines train and validation sets for training
        early_stopping: If True, enables early stopping based on validation loss
        patience: Number of epochs to wait before early stopping after validation loss increases

    Returns:
        model, metrics: Trained model and training metrics
    """
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    # Load datasets
    train_dataset = PathVQAYesNoDataset(split='train', num_samples=num_samples)

    if use_combined_data:
        # Load validation set and combine with training set
        val_dataset = PathVQAYesNoDataset(split='validation',
                                          num_samples=num_samples)
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_dataset, val_dataset])
        log(f"Using combined train+validation dataset with {len(combined_dataset)} samples")

        # Use combined dataset for training
        train_loader = DataLoader(combined_dataset, batch_size=batch_size,
                                  shuffle=True)

        # Create a small held-out validation set (10% of training data) for tracking metrics
        from torch.utils.data import random_split
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        _, small_val_dataset = random_split(train_dataset,
                                            [train_size, val_size])
        val_loader = DataLoader(small_val_dataset, batch_size=batch_size)
        log(f"Created small validation set with {len(small_val_dataset)} samples for tracking metrics")
    else:
        # Standard training with separate validation set
        val_dataset = PathVQAYesNoDataset(split='validation',
                                          num_samples=num_samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if use_combined_data:
        log(f"Training with {len(combined_dataset)} combined samples (train + validation)")
    else:
        log(f"Training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    log(f"Hyperparameters: lr={lr}, batch_size={batch_size}, dropout={dropout}, weight_decay={weight_decay}")

    model = PathVQAYesNoModel(dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCELoss()

    metrics = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [],
        'val_acc': []
    }

    # Training loop
    for epoch in range(num_epochs):
        log(f"Epoch {epoch + 1}/{num_epochs}:")
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].float().unsqueeze(1).to(DEVICE)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)

        # Use the appropriate dataset length based on which mode we're in
        if use_combined_data:
            train_acc = correct / len(combined_dataset)
        else:
            train_acc = correct / len(train_dataset)

        log(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion)
        log(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        # Early stopping check (if enabled)
        if early_stopping and len(metrics["val_loss"]) >= patience:
            # Check if validation loss has been increasing for 'patience' epochs
            increasing = True
            for i in range(1, patience):
                if metrics["val_loss"][-i] <= metrics["val_loss"][-i-1]:
                    increasing = False
                    break
            
            if increasing:
                log(f"Early stopping: validation loss increased {patience} times in a row.")
                break

    # Save model and metrics if requested
    if save_model:
        sample_count = len(
            train_dataset) if num_samples is None else num_samples
        sample_label = "All" if num_samples is None else str(sample_count)

        # Use custom path if provided, otherwise use default path
        if custom_model_path:
            model_path = custom_model_path
        else:
            # Add suffix to indicate combined data usage
            combined_suffix = "_combined" if use_combined_data else ""
            model_path = os.path.join(get_project_path("models"),
                                      f"pathvqa_yesno_model_s{sample_label}{combined_suffix}_b{batch_size}.pt")

        torch.save(model.state_dict(), model_path)
        log(f"Model saved to {model_path}")

        # Save metrics with matching filename
        metrics_filename = os.path.basename(model_path).replace('.pt', '.json')
        metrics_path = os.path.join(get_project_path("metrics"),
                                    metrics_filename)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        log(f"Training metrics saved to {metrics_path}")

    return model, metrics


def plot_training_metrics(metrics, num_epochs, use_combined_data=False):
    """
    Plot the training and validation metrics over extended epochs.
    
    Args:
        metrics: Dictionary containing training metrics
        num_epochs: Number of epochs trained
        use_combined_data: Whether combined data was used
    """
    plt.figure(figsize=(12, 10))

    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-',
             label='Training Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], 'r-',
             label='Validation Loss')
    plt.title(f'Loss over {num_epochs} Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epoch'], metrics['train_acc'], 'b-',
             label='Training Accuracy')
    plt.plot(metrics['epoch'], metrics['val_acc'], 'r-',
             label='Validation Accuracy')
    plt.title(f'Accuracy over {num_epochs} Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plot
    combined_suffix = "_combined" if use_combined_data else ""
    plot_path = os.path.join(get_project_path("plots"),
                             f"full_training{combined_suffix}_{num_epochs}epochs.png")
    plt.savefig(plot_path)
    print(f"Full training plot saved to: {plot_path}")


def train_with_best_params(num_epochs=20, use_combined_data=False, early_stopping=True, patience=5):
    """
    Train a model with the best parameters from optuna.
    
    Args:
        num_epochs: Number of epochs to train (default: 20)
        use_combined_data: Whether to use combined train+validation data
        early_stopping: Whether to enable early stopping
        patience: Number of epochs to wait before early stopping (default: 5)
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"full_training_{timestamp}"
    logger, log_file = setup_logger(run_name)

    logger.info("=" * 50)
    logger.info(f"Full Training with Best Parameters for {num_epochs} epochs")
    logger.info("=" * 50)

    # Load best parameters from the saved file
    best_params_path = os.path.join(get_project_path("models"),
                                    "best_params_sAll.json")
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    logger.info(f"Loaded best parameters: {json.dumps(best_params, indent=2)}")

    # Create output paths
    combined_suffix = "_combined" if use_combined_data else ""
    model_path = os.path.join(get_project_path("models"),
                              f"pathvqa_yesno_model_full{combined_suffix}_{num_epochs}epochs.pt")

    # Train model with best parameters
    model, metrics = train_with_params(
        num_samples=None,  # Use all available samples
        batch_size=best_params['batch_size'],
        num_epochs=num_epochs,
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        dropout=best_params['dropout'],
        save_model=True,
        custom_model_path=model_path,
        logger=logger,
        use_combined_data=use_combined_data,
        early_stopping=early_stopping,
        patience=patience
    )

    # Plot full training metrics
    plot_training_metrics(metrics, num_epochs, use_combined_data)

    logger.info("\n" + "=" * 50)
    logger.info("Full training with best parameters completed successfully!")
    logger.info("=" * 50)
    logger.info(f"Full logs saved to: {log_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PathVQA Yes/No model")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of training samples to use (None=all)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument("--use-best-params", action="store_true",
                        help="Use best parameters from optuna")
    parser.add_argument("--full-epochs", type=int, default=20,
                        help="Number of epochs when training with best parameters (default: 20)")
    parser.add_argument("--combined-data", action="store_true",
                        help="Use combined training and validation data")
    parser.add_argument("--disable-early-stopping", action="store_true",
                        help="Disable early stopping and train for full epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait before early stopping (default: 5)")

    args = parser.parse_args()

    if args.use_best_params:
        # Run training with best parameters from optuna
        train_with_best_params(
            num_epochs=args.full_epochs, 
            use_combined_data=args.combined_data,
            early_stopping=not args.disable_early_stopping,
            patience=args.patience
        )
    else:
        # Run regular training with specified parameters
        model, metrics = train_with_params(
            num_samples=args.samples if args.samples != 0 else None,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            use_combined_data=args.combined_data
        )
        print("Training completed successfully")
