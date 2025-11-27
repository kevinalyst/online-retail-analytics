#!/usr/bin/env python3
"""
Customer Lifetime Value (CLV) Model Training Script

This script trains a gradient boosting regression model to predict customer lifetime value
based on historical transaction features. It performs data preprocessing, model training,
evaluation, and generates predictions for all customers.

Author: ML Engineering Team
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_input_file(path: str) -> None:
    """
    Validate that the input file exists and is readable.
    
    Parameters
    ----------
    path : str
        Path to the input CSV file
        
    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info(f"Input file validated: {path}")


def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess the CLV features dataset.
    
    This function performs the following steps:
    1. Loads the CSV file
    2. Maps column names to expected format (handles variations)
    3. Drops rows with missing CustomerID or target variable
    4. Ensures feature columns are numeric
    5. Imputes missing values with median
    6. Clips extreme outliers in target variable
    
    Parameters
    ----------
    path : str
        Path to the input CSV file
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        - X: Feature DataFrame with 5 numerical columns
        - y: Target Series (future_6m_revenue)
        - customer_ids: Series of customer identifiers
        
    Raises
    ------
    ValueError
        If required columns are missing from the dataset
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Define column mapping to handle naming variations
    column_mapping = {
        'customer_id': 'CustomerID',
        'total_orders': 'hist_orders',
        'total_revenue': 'hist_revenue',
        'avg_order_value': 'hist_avg_order_value',
        'days_since_first': 'days_since_first',
        'days_since_last': 'days_since_last',
        'future_6M_revenue': 'future_6m_revenue'
    }
    
    # Apply column mapping (case-insensitive)
    df_columns_lower = {col.lower(): col for col in df.columns}
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name.lower() in df_columns_lower:
            rename_dict[df_columns_lower[old_name.lower()]] = new_name
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Mapped columns: {rename_dict}")
    
    # Expected columns
    required_cols = ['CustomerID', 'hist_orders', 'hist_revenue', 'hist_avg_order_value',
                     'days_since_first', 'days_since_last', 'future_6m_revenue']
    
    # Check for missing required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Store original row count
    original_count = len(df)
    
    # Drop rows with missing CustomerID or target variable
    df = df.dropna(subset=['CustomerID', 'future_6m_revenue'])
    dropped_count = original_count - len(df)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with missing CustomerID or future_6m_revenue")
    
    # Extract customer IDs
    customer_ids = df['CustomerID'].copy()
    
    # Define feature columns
    feature_cols = ['hist_orders', 'hist_revenue', 'hist_avg_order_value',
                    'days_since_first', 'days_since_last']
    
    # Ensure feature columns are numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values in features with median imputation
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Imputed {missing_count} missing values in {col} with median={median_val:.2f}")
    
    # Clip extreme outliers in target variable for robustness
    target_99 = df['future_6m_revenue'].quantile(0.99)
    outliers_count = (df['future_6m_revenue'] > target_99).sum()
    if outliers_count > 0:
        logger.info(f"Clipping {outliers_count} extreme outliers in target (>99th percentile: {target_99:.2f})")
        df['future_6m_revenue'] = df['future_6m_revenue'].clip(upper=target_99)
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['future_6m_revenue'].copy()
    
    logger.info(f"Preprocessed data shape - X: {X.shape}, y: {y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    logger.info(f"Target statistics - mean: {y.mean():.2f}, std: {y.std():.2f}, min: {y.min():.2f}, max: {y.max():.2f}")
    
    return X, y, customer_ids


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """
    Train a gradient boosting regression model with hyperparameter tuning.
    
    This function uses RandomizedSearchCV to find the best hyperparameters
    for a GradientBoostingRegressor model, optimizing for negative RMSE.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
        
    Returns
    -------
    GradientBoostingRegressor
        The best estimator found by RandomizedSearchCV
    """
    logger.info("Starting model training with hyperparameter search")
    
    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Initialize base model
    base_model = GradientBoostingRegressor(random_state=42, loss='squared_error')
    
    # Setup randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',  # Optimize for RMSE
        n_jobs=-1,  # Use all available cores
        random_state=42,
        verbose=1
    )
    
    logger.info(f"Training with {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Log best parameters and score
    logger.info(f"Best hyperparameters: {random_search.best_params_}")
    best_rmse = np.sqrt(-random_search.best_score_)
    logger.info(f"Best CV RMSE: {best_rmse:.2f}")
    
    return random_search.best_estimator_


def evaluate_model(model: GradientBoostingRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the trained model on the test set.
    
    Computes and logs the following metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    
    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    """
    logger.info("Evaluating model on test set")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log results
    logger.info("=" * 50)
    logger.info("TEST SET EVALUATION METRICS")
    logger.info("=" * 50)
    logger.info(f"Test RMSE: {rmse:.2f}")
    logger.info(f"Test MAE:  {mae:.2f}")
    logger.info(f"Test R²:   {r2:.4f}")
    logger.info("=" * 50)


def train_final_model(model: GradientBoostingRegressor, X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    """
    Train the final model on all available data.
    
    This function takes the best model configuration found during hyperparameter
    search and refits it on the entire dataset to maximize prediction quality.
    
    Parameters
    ----------
    model : GradientBoostingRegressor
        Best model from hyperparameter search (used for configuration)
    X : pd.DataFrame
        All available features
    y : pd.Series
        All available target values
        
    Returns
    -------
    GradientBoostingRegressor
        Model trained on all data
    """
    logger.info("Training final model on all data")
    logger.info(f"Final training set size: {X.shape[0]} samples")
    
    # Create a new model with the same parameters as the best model
    final_model = GradientBoostingRegressor(**model.get_params())
    final_model.fit(X, y)
    
    logger.info("Final model training complete")
    return final_model


def generate_predictions(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
    customer_ids: pd.Series,
    y_actual: pd.Series = None
) -> pd.DataFrame:
    """
    Generate CLV predictions for all customers.
    
    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained model
    X : pd.DataFrame
        Features for all customers
    customer_ids : pd.Series
        Customer identifiers
    y_actual : pd.Series, optional
        Actual future revenue values for comparison
        
    Returns
    -------
    pd.DataFrame
        DataFrame with CustomerID, clv_prediction, and optionally actual values
    """
    logger.info("Generating predictions for all customers")
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'CustomerID': customer_ids.values,
        'clv_prediction': predictions
    })
    
    # Optionally add actual values for comparison
    if y_actual is not None:
        output_df['actual_future_6m_revenue'] = y_actual.values
    
    logger.info(f"Generated {len(output_df)} predictions")
    logger.info(f"Prediction statistics - mean: {predictions.mean():.2f}, "
                f"std: {predictions.std():.2f}, min: {predictions.min():.2f}, "
                f"max: {predictions.max():.2f}")
    
    return output_df


def save_model(model: GradientBoostingRegressor, path: str) -> None:
    """
    Save the trained model to disk using joblib.
    
    Parameters
    ----------
    model : GradientBoostingRegressor
        Trained model to save
    path : str
        Output path for the model file
    """
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)
    logger.info(f"Model saved successfully")


def save_predictions(predictions_df: pd.DataFrame, path: str) -> None:
    """
    Save predictions to a CSV file.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions
    path : str
        Output path for the CSV file
    """
    logger.info(f"Saving predictions to {path}")
    predictions_df.to_csv(path, index=False)
    logger.info(f"Predictions saved successfully ({len(predictions_df)} rows)")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a CLV prediction model using gradient boosting regression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        default='clv_features.csv',
        help='Path to input CSV file with CLV features'
    )
    
    parser.add_argument(
        '--output-pred-path',
        type=str,
        default='customer_clv_pred.csv',
        help='Path to output CSV file with CLV predictions'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='clv_model.joblib',
        help='Path to save the trained model'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate test_size
    if not 0.0 < args.test_size < 1.0:
        parser.error("test-size must be between 0.0 and 1.0")
    
    return args


def main() -> None:
    """
    Main execution function for CLV model training pipeline.
    
    This function orchestrates the entire workflow:
    1. Parse command-line arguments
    2. Load and preprocess data
    3. Split into train/test sets
    4. Train model with hyperparameter tuning
    5. Evaluate on test set
    6. Train final model on all data
    7. Generate and save predictions
    8. Save trained model
    """
    try:
        # Parse arguments
        args = parse_args()
        
        logger.info("=" * 70)
        logger.info("CLV MODEL TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Configuration:")
        logger.info(f"  Input path:       {args.input_path}")
        logger.info(f"  Output pred path: {args.output_pred_path}")
        logger.info(f"  Model path:       {args.model_path}")
        logger.info(f"  Test size:        {args.test_size}")
        logger.info(f"  Random state:     {args.random_state}")
        logger.info("=" * 70)
        
        # Validate input file
        validate_input_file(args.input_path)
        
        # Load and preprocess data
        X, y, customer_ids = load_and_preprocess(args.input_path)
        
        # Split data into train and test sets
        logger.info(f"Splitting data: {(1-args.test_size)*100:.0f}% train, {args.test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set:  {X_test.shape[0]} samples")
        
        # Train model with hyperparameter search
        best_model = train_model(X_train, y_train)
        
        # Evaluate model on test set
        evaluate_model(best_model, X_test, y_test)
        
        # Train final model on all data
        final_model = train_final_model(best_model, X, y)
        
        # Generate predictions for all customers
        predictions_df = generate_predictions(final_model, X, customer_ids, y)
        
        # Save predictions
        save_predictions(predictions_df, args.output_pred_path)
        
        # Save trained model
        save_model(final_model, args.model_path)
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Outputs:")
        logger.info(f"  Predictions: {args.output_pred_path}")
        logger.info(f"  Model:       {args.model_path}")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
