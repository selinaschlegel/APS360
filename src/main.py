import sys
from datetime import datetime
import os
import matplotlib

from logger import setup_logger, get_project_path
from optuna_hp import find_best_hyperparams
from test_model import evaluate_model, plot_accuracy_comparison, plot_metrics
from train_model import train_with_params

# Use non-interactive backend to prevent hanging
matplotlib.use('Agg')


def main(num_samples=None, optuna_trials=20, use_combined_data=False):  # None means use all available samples
    try:
        # Set up logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_label = "All" if num_samples is None else str(num_samples)
        run_name = f"pathvqa_s{sample_label}_{timestamp}"
        logger, log_file = setup_logger(run_name)
        
        logger.info("=" * 50)
        logger.info("PathVQA Yes/No Model Training Pipeline")
        logger.info("=" * 50)
        logger.info(f"Logging to: {log_file}")

        # First run Optuna to find the best hyperparameters
        logger.info("\nStep 1: Finding optimal hyperparameters with Optuna...")
        best_params = find_best_hyperparams(num_samples, optuna_trials=optuna_trials, logger=logger)

        # Get the actual sample count from the dataset
        from train_model import PathVQAYesNoDataset
        train_dataset = PathVQAYesNoDataset(split='train',
                                            num_samples=num_samples)
        actual_sample_count = len(train_dataset)
        sample_label = "All" if num_samples is None else str(actual_sample_count)

        # Train the final model with the best hyperparameters
        if use_combined_data:
            logger.info(f"\nStep 2: Training final model with COMBINED training and validation data...")
        else:
            logger.info(f"\nStep 2: Training final model with {actual_sample_count} samples and best hyperparameters...")
        
        # Add suffix for combined data if used
        combined_suffix = "_combined" if use_combined_data else ""
        model_path = os.path.join(get_project_path("models"), f"pathvqa_yesno_model_s{sample_label}{combined_suffix}.pt")
        
        model, metrics = train_with_params(
            num_samples=num_samples,
            batch_size=best_params['batch_size'],
            num_epochs=10,
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            dropout=best_params['dropout'],
            save_model=True,
            custom_model_path=model_path,
            logger=logger,
            use_combined_data=use_combined_data
        )

        # Evaluate the model
        logger.info("\nStep 3: Evaluating final model...")
        test_metrics = evaluate_model(model_path, logger=logger)
        test_acc = test_metrics['accuracy']

        # Plot results
        logger.info("\nStep 4: Generating performance visualizations...")

        # Plot accuracy comparison
        combined_suffix = "_combined" if use_combined_data else ""
        accuracy_plot_path = os.path.join(get_project_path("plots"), f"accuracy_s{sample_label}{combined_suffix}.png")
        plot_accuracy_comparison(
            model_path,
            test_acc=test_metrics,
            save_path=accuracy_plot_path,
            show_plot=False
        )
        logger.info(f"Accuracy plot saved to {accuracy_plot_path}")

        # Plot metrics visualization
        metrics_plot_path = os.path.join(get_project_path("plots"), f"metrics_s{sample_label}{combined_suffix}.png")
        plot_metrics(
            model_path,
            test_metrics=test_metrics,
            save_path=metrics_plot_path,
            show_plot=False
        )
        logger.info(f"Metrics plot saved to {metrics_plot_path}")

        logger.info("\n" + "=" * 50)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 50)
        logger.info(f"Full logs saved to: {log_file}")

        sys.exit(0)
    except Exception as e:
        logger.error(f"\nError occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate PathVQA Yes/No model")
    parser.add_argument("--samples", type=int, default=None, 
                        help="Number of training samples to use (None=all)")
    parser.add_argument("--trials", type=int, default=20, 
                        help="Number of Optuna trials for hyperparameter search")
    parser.add_argument("--combined-data", action="store_true",
                        help="Use combined training and validation data for final model training")
    
    args = parser.parse_args()
    
    main(num_samples=args.samples, optuna_trials=args.trials, use_combined_data=args.combined_data)
