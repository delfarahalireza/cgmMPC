#!/usr/bin/env python3
"""
FT Predictor Training
Implements patient-by-patient forward chaining validation as described in Aiello et al. 2023
Forward chaining: Train on patients 1 to k_p, validate on patient k_p+1
8-step ahead predictions (2 hours at 15-min intervals)
Ground truth vs prediction assessment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Dict
import pickle
import warnings
warnings.filterwarnings('ignore')

class SequenceDataset(Dataset):
    """Dataset for LSTM sequence prediction"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FTPredictor(nn.Module):
    """
    FT Predictor LSTM
    Input: Complete state vector (CGM + insulin + carbs + time)
    Output: 8-step glucose predictions (2 hours ahead)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, prediction_horizon: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, prediction_horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length=1, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Generate predictions for next 8 steps
        predictions = self.output_layer(last_output)  # (batch_size, prediction_horizon)
        
        return predictions

class FTSystem:
    """FT training system with patient-by-patient forward chaining"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 prediction_horizon: int = 8, learning_rate: float = 0.001):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = FTPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon
        )
        
        # Training components
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Loss function - MAE for glucose prediction
        self.criterion = nn.L1Loss()
        
        # Scalers
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.is_fitted = False
        
        # Training history for each iteration
        self.iteration_results = []
        
    def prepare_patient_data(self, data_dict):
        """Prepare data organized by patients for forward chaining"""
        
        print("Preparing patient-organized data...")
        
        # Combine training and validation data
        all_X = np.vstack([data_dict['training']['scenario_1']['X'], 
                          data_dict['validation']['scenario_1']['X']])
        all_y = np.vstack([data_dict['training']['scenario_1']['y'], 
                          data_dict['validation']['scenario_1']['y']])
        all_patient_ids = np.hstack([data_dict['training']['scenario_1']['patient_ids'],
                                   data_dict['validation']['scenario_1']['patient_ids']])
        
        # Organize by patient
        patient_data = {}
        unique_patients = sorted(np.unique(all_patient_ids))
        
        for patient_id in unique_patients:
            mask = all_patient_ids == patient_id
            patient_data[patient_id] = {
                'X': all_X[mask],
                'y': all_y[mask]
            }
        
        print(f"Organized data for {len(unique_patients)} patients: {unique_patients}")
        for patient_id in unique_patients:
            print(f"  Patient {patient_id}: {len(patient_data[patient_id]['X'])} samples")
        
        return patient_data, unique_patients
    
    def forward_chaining_train(self, patient_data: Dict, unique_patients: List[int],
                             epochs_per_iteration: int = 100, batch_size: int = 64,
                             early_stopping_patience: int = 20, verbose: bool = True):
        """
        Implement forward chaining methodology
        At iteration k_p: Train on patients 1 to k_p, validate on patient k_p+1
        """
        
        print("FORWARD CHAINING TRAINING")
        print("=" * 60)
        print("Methodology: Train on patients 1 to k_p, validate on patient k_p+1")
        print(f"Total iterations: {len(unique_patients) - 1}")
        
        self.iteration_results = []
        
        for k_p in range(1, len(unique_patients)):
            print(f"\nITERATION {k_p}/{len(unique_patients)-1}")
            print("-" * 40)
            
            # Training patients: 0 to k_p-1 (inclusive)
            train_patients = unique_patients[:k_p]
            # Validation patient: k_p
            val_patient = unique_patients[k_p]
            
            print(f"Training patients: {train_patients}")
            print(f"Validation patient: {val_patient}")
            
            # Prepare training data
            X_train_list = []
            y_train_list = []
            for patient_id in train_patients:
                X_train_list.append(patient_data[patient_id]['X'])
                y_train_list.append(patient_data[patient_id]['y'])
            
            X_train = np.vstack(X_train_list)
            y_train = np.vstack(y_train_list)
            
            # Validation data
            X_val = patient_data[val_patient]['X']
            y_val = patient_data[val_patient]['y']
            
            print(f"Training samples: {len(X_train):,}")
            print(f"Validation samples: {len(X_val):,}")
            
            # Fit scalers on training data only
            self.scaler_X.fit(X_train)
            self.scaler_y.fit(y_train)
            
            # Scale data
            X_train_scaled = self.scaler_X.transform(X_train)
            y_train_scaled = self.scaler_y.transform(y_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            
            # Reshape for LSTM (add sequence dimension)
            X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_val_seq = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
            
            # Create datasets
            train_dataset = SequenceDataset(X_train_seq, y_train_scaled)
            val_dataset = SequenceDataset(X_val_seq, y_val_scaled)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Reinitialize model for each iteration (as per Aiello et al. methodology)
            self.model = FTPredictor(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                prediction_horizon=self.prediction_horizon
            ).to(self.device)
            
            # Reinitialize optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop for this iteration
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs_per_iteration):
                # Training phase
                self.model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Progress logging
                if verbose and (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1:3d}/{epochs_per_iteration}] | '
                          f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
                          f'Best: {best_val_loss:.4f}')
            
            # Load best model for this iteration
            if 'best_model_state' in locals():
                self.model.load_state_dict(best_model_state)
            
            # Evaluate on validation patient (8-step predictions)
            iteration_result = self._evaluate_iteration(
                X_val, y_val, val_patient, k_p, train_patients
            )
            
            iteration_result.update({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'total_epochs': len(train_losses)
            })
            
            self.iteration_results.append(iteration_result)
            
            print(f"Iteration {k_p} completed:")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            print(f"  8-step MAE: {iteration_result['mae_8_step']:.2f} mg/dL")
            print(f"  15-min MAE: {iteration_result['mae_15_min']:.2f} mg/dL")
        
        self.is_fitted = True
        
        # Save final model (from last iteration)
        torch.save(self.model.state_dict(), 'ft_predictor.pth')
        print(f"\nFinal model saved as 'ft_predictor.pth'")
        
        return self.iteration_results
    
    def _evaluate_iteration(self, X_val: np.ndarray, y_val: np.ndarray, 
                          val_patient: int, iteration: int, train_patients: List[int]) -> Dict:
        """Evaluate model performance for this iteration"""
        
        # Scale validation data
        X_val_scaled = self.scaler_X.transform(X_val)
        X_val_seq = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse scale predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Calculate metrics
        # 15-minute ahead (step 1)
        mae_15_min = np.mean(np.abs(predictions[:, 0] - y_val[:, 0]))
        
        # 8-step ahead (2 hours, step 8)
        mae_8_step = np.mean(np.abs(predictions[:, 7] - y_val[:, 7]))
        
        # Overall MAE across all steps
        mae_overall = np.mean(np.abs(predictions - y_val))
        
        # Step-wise MAE
        step_wise_mae = [np.mean(np.abs(predictions[:, i] - y_val[:, i])) 
                        for i in range(self.prediction_horizon)]
        
        return {
            'iteration': iteration,
            'val_patient': val_patient,
            'train_patients': train_patients,
            'mae_15_min': mae_15_min,
            'mae_8_step': mae_8_step,
            'mae_overall': mae_overall,
            'step_wise_mae': step_wise_mae,
            'predictions': predictions,
            'ground_truth': y_val,
            'num_samples': len(y_val)
        }
    
    def plot_24hour_population_average(self, patient_data: Dict, unique_patients: List[int]):
        """Plot 24-hour population average ground truth vs predicted"""
        
        if not self.iteration_results:
            print("No iteration results to plot")
            return
        
        print(f"\nCreating 24-hour population average plot...")
        
        # Get all available data and predictions from final model
        final_result = self.iteration_results[-1]
        
        # Collect data from all patients and get their predictions
        all_ground_truth = []
        all_predictions = []
        all_timestamps = []
        
        # Load original data to get timestamps
        with open('cache/temporal_simglucose_data.pkl', 'rb') as f:
            original_data = pickle.load(f)
        
        # Get time features from the original data
        all_X = np.vstack([original_data['training']['scenario_1']['X'], 
                          original_data['validation']['scenario_1']['X']])
        all_patient_ids = np.hstack([original_data['training']['scenario_1']['patient_ids'],
                                   original_data['validation']['scenario_1']['patient_ids']])
        time_features = all_X[:, -1]  # Last column is time of day
        
        # Collect predictions for all patients using the final trained model
        for patient_id in unique_patients:
            mask = all_patient_ids == patient_id
            patient_X = all_X[mask]
            patient_time = time_features[mask]
            patient_ground_truth = patient_data[patient_id]['y']
            
            # Get predictions for this patient
            patient_predictions = self.predict(patient_X)
            
            all_ground_truth.append(patient_ground_truth)
            all_predictions.append(patient_predictions)
            all_timestamps.extend(patient_time)
        
        # Combine all data
        all_ground_truth = np.vstack(all_ground_truth)
        all_predictions = np.vstack(all_predictions)
        all_timestamps = np.array(all_timestamps)
        
        # Create time bins (30-minute intervals for smoother curves)
        time_bins = np.arange(0, 24, 0.5)
        time_centers = time_bins[:-1] + 0.25
        
        # Calculate population averages for each time bin and each prediction step
        n_steps = 8
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Define colors for different prediction horizons
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
        
        # Plot ground truth (single line)
        gt_profile = []
        for i in range(len(time_bins) - 1):
            bin_mask = (all_timestamps >= time_bins[i]) & (all_timestamps < time_bins[i + 1])
            if np.any(bin_mask):
                # Use current glucose (step 0 of predictions represents current state)
                gt_avg = np.mean(all_ground_truth[bin_mask, 0])
                gt_profile.append(gt_avg)
            else:
                gt_profile.append(np.nan)
        
        # Plot ground truth
        ax.plot(time_centers, gt_profile, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
        
        # Plot predictions for each step
        selected_steps = [0, 1, 3, 7]  # 15min, 30min, 1h, 2h
        step_labels = ['15 min', '30 min', '1 hour', '2 hours']
        
        for i, step in enumerate(selected_steps):
            pred_profile = []
            for j in range(len(time_bins) - 1):
                bin_mask = (all_timestamps >= time_bins[j]) & (all_timestamps < time_bins[j + 1])
                if np.any(bin_mask):
                    pred_avg = np.mean(all_predictions[bin_mask, step])
                    pred_profile.append(pred_avg)
                else:
                    pred_profile.append(np.nan)
            
            ax.plot(time_centers, pred_profile, '--', linewidth=2, 
                   color=colors[i], label=f'LSTM {step_labels[i]} ahead', alpha=0.8)
        
        # Add meal time indicators
        meal_times = [8, 13, 19]
        for meal_time in meal_times:
            ax.axvline(x=meal_time, color='red', linestyle=':', alpha=0.6, linewidth=2)
        
        # Formatting
        ax.set_xlabel('Time of Day (hours)', fontsize=12)
        ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        ax.set_title('24-Hour Population Average: Ground Truth vs LSTM Predictions', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        # Add meal annotations
        meal_names = ['Breakfast', 'Lunch', 'Dinner']
        for i, (meal_time, meal_name) in enumerate(zip(meal_times, meal_names)):
            ax.annotate(meal_name, xy=(meal_time, ax.get_ylim()[1] * 0.95), 
                       xytext=(meal_time, ax.get_ylim()[1] * 0.98),
                       ha='center', va='bottom', fontsize=9, color='red',
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
        
        plt.tight_layout()
        
        # Save plot
        filename = 'ft_population_average.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 24-hour population average plot saved as '{filename}'")
        
        # Print statistics
        print(f"\nPopulation Average Statistics:")
        for i, step in enumerate(selected_steps):
            step_mae = np.mean(np.abs(all_predictions[:, step] - all_ground_truth[:, step]))
            print(f"  {step_labels[i]} ahead MAE: {step_mae:.2f} mg/dL")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input
        X_scaled = self.scaler_X.transform(X)
        X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions

def main():
    """Main FT training function"""
    
    print("FT PREDICTOR TRAINING")
    print("=" * 60)
    print("Implementation:")
    print("Patient-by-patient forward chaining validation") 
    print("8-step ahead predictions (2 hours)")
    print("Cross-patient generalization assessment")
    print("Ground truth vs prediction plotting")
    print("=" * 60)
    
    # Load data
    try:
        with open('cache/temporal_simglucose_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Loaded SimGlucose temporal data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize FT system
    input_size = data['training']['scenario_1']['X'].shape[1]
    
    ft = FTSystem(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        prediction_horizon=8,
        learning_rate=0.001
    )
    
    # Prepare patient-organized data
    patient_data, unique_patients = ft.prepare_patient_data(data)
    
    # Run forward chaining training
    results = ft.forward_chaining_train(
        patient_data, unique_patients,
        epochs_per_iteration=100,
        batch_size=64,
        early_stopping_patience=20,
        verbose=True
    )

    
    # Plot 24-hour population average
    ft.plot_24hour_population_average(patient_data, unique_patients)
    
    print("\\n" + "=" * 60)
    print("FT TRAINING COMPLETED")
    print("=" * 60)
    print("Patient-by-patient forward chaining completed")
    print("Cross-patient generalization evaluated")
    print("Ready for GT predictor training")
    print("=" * 60)

if __name__ == "__main__":
    main()