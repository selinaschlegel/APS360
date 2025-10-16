import os
import json
import optuna
from train_model import train_with_params
from logger import get_project_path


def find_best_hyperparams(num_samples, optuna_trials=5, optuna_epochs=5,
                          use_full_data_for_optuna=False, optuna_samples=3000,
                          logger=None):
    """
    Uses Optuna to find the best hyperparameters for the model

    Args:
        num_samples: Number of samples for final training
        optuna_trials: Number of Optuna trials to run
        optuna_epochs: Number of epochs to train each trial model
        use_full_data_for_optuna: If True, uses the same sample size for Optuna as final training
        optuna_samples: Number of samples to use during Optuna optimization (if use_full_data_for_optuna is False)
        logger: Logger instance for recording output

    Returns:
        best_params: Dictionary of best hyperparameters
    """
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print
    
    log(f"Starting hyperparameter optimization with Optuna ({optuna_trials} trials)")

    # Set the sample size for Optuna
    samples_for_optuna = num_samples if use_full_data_for_optuna else optuna_samples
    log(f"Using {samples_for_optuna} samples for hyperparameter optimization")

    # Define the Optuna objective function
    def objective(trial):
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3,
                                           log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        log(f"\nTrial {trial.number + 1}/{optuna_trials}:")
        log(f"Testing: dropout={dropout:.3f}, lr={lr:.6f}, weight_decay={weight_decay:.6f}, batch_size={batch_size}")

        # Train the model with these hyperparameters but don't save it
        _, metrics = train_with_params(
            num_samples=samples_for_optuna,
            batch_size=batch_size,
            num_epochs=optuna_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            save_model=False,
            logger=logger
        )

        # Return the validation accuracy as the optimization target
        val_acc = metrics['val_acc'][-1]
        log(f"Trial result: validation accuracy = {val_acc:.4f}")
        return val_acc

    # Create and run the Optuna study
    study = optuna.create_study(direction="maximize")
    log("\nStarting Optuna trials...")
    log("=" * 50)

    try:
        study.optimize(objective, n_trials=optuna_trials)
    except KeyboardInterrupt:
        log("\nOptuna optimization was interrupted. "
            "Using best parameters found so far.")

    log("\n" + "=" * 50)
    log(f"Optuna completed {len(study.trials)} trials.")

    # Get the best parameters
    best_params = study.best_trial.params
    best_accuracy = study.best_trial.value
    log("\nBest hyperparameters found:")
    log(json.dumps(best_params, indent=2))
    log(f"Best validation accuracy: {best_accuracy:.4f}")

    # Save the best parameters to file
    sample_label = "All" if num_samples is None else str(num_samples)
    params_path = os.path.join(get_project_path("models"), f"best_params_s{sample_label}.json")
    with open(params_path, 'w') as f:
        json.dump(best_params, f)
    log(f"Best parameters saved to {params_path}")

    return best_params


if __name__ == "__main__":
    # Example of standalone usage
    best_params = find_best_hyperparams(5000, optuna_trials=3)
    print("Hyperparameter optimization completed")