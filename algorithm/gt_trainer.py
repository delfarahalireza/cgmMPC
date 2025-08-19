#!/usr/bin/env python3
"""
GT Predictor Training
1. Use only CGM input features (subset of state vector)
2. Target FT correction factors instead of direct differences  
3. Learn insulin effects indirectly from glucose patterns

GT learns: ΔYcgm,FT = GT · ΔU (indirectly through CGM patterns)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class GTDataset(Dataset):
    """Dataset for GT predictor training"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GTPredictor(nn.Module):
    """
    GT Predictor
    Input: Only CGM data (past glucose trends)
    Output: Correction factors for FT predictions
    """
    
    def __init__(self, cgm_history_length: int = 25, hidden_size: int = 64,
                 num_layers: int = 2, prediction_horizon: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.cgm_history_length = cgm_history_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM for processing CGM history only
        self.lstm = nn.LSTM(
            input_size=1,  # Single CGM value at each timestep
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output network for correction factors
        self.correction_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, prediction_horizon)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, cgm_sequence):
        # cgm_sequence shape: (batch_size, sequence_length, 1)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(cgm_sequence)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply layer normalization
        last_hidden = self.layer_norm(last_hidden)
        
        # Generate correction factors
        corrections = self.correction_network(last_hidden)
        
        return corrections

class GTSystem:
    """
    GT Predictor System
    Learns insulin effects indirectly through CGM patterns
    """
    
    def __init__(self, cgm_history_length: int = 25, hidden_size: int = 64,
                 num_layers: int = 2, prediction_horizon: int = 8,
                 learning_rate: float = 0.001):
        
        self.cgm_history_length = cgm_history_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        
        # Initialize GT model
        self.model = GTPredictor(
            cgm_history_length=cgm_history_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon,
            dropout=0.1
        )
        
        # Training components
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=2, eta_min=1e-6
        )
        
        # Loss function for correction factors
        self.criterion = nn.MSELoss()  # Simple MSE for correction factors
        
        # Scalers for CGM input and correction targets
        self.scaler_cgm = RobustScaler()
        self.scaler_corrections = RobustScaler()
        self.is_fitted = False
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def prepare_gt_data(self, simglucose_data, ft_system):
        """
        Prepare GT training data
        """
        
        print("Preparing GT training data...")
        print("1. Use only CGM input features")
        print("2. Target FT correction factors")
        print("3. Learn insulin effects indirectly from glucose patterns")
        
        # Get data from both scenarios
        scenario_1_data = simglucose_data['training']['scenario_1']
        scenario_2_data = simglucose_data['training']['scenario_2']
        
        X_s1 = scenario_1_data['X']
        y_s1 = scenario_1_data['y']
        y_s2 = scenario_2_data['y']
        
        print(f"Scenario-I data: {X_s1.shape}")
        print(f"Scenario-II targets: {y_s2.shape}")
        
        # Step 1: Extract only CGM features (first 25 features are CGM history)
        cgm_features = X_s1[:, :self.cgm_history_length]
        print(f"Extracted CGM features: {cgm_features.shape}")
        
        # Step 2: Get FT predictions on Scenario-I data
        print("\\nGenerating FT predictions for correction calculation...")
        ft_predictions = ft_system.predict(X_s1)
        print(f"FT predictions shape: {ft_predictions.shape}")
        
        # Step 3: Calculate correction factors needed
        # Correction = Actual_Scenario_II - FT_Prediction_Scenario_I
        min_samples = min(len(ft_predictions), len(y_s2))
        correction_factors = y_s2[:min_samples] - ft_predictions[:min_samples]
        cgm_features_aligned = cgm_features[:min_samples]
        
        print(f"Correction factors shape: {correction_factors.shape}")
        print(f"Mean correction magnitude: {np.mean(np.abs(correction_factors)):.2f} mg/dL")
        print(f"Correction range: {np.min(correction_factors):.1f} to {np.max(correction_factors):.1f} mg/dL")
        
        # Step 4: Reshape CGM features for LSTM input
        # From (samples, cgm_history) to (samples, cgm_history, 1)
        cgm_sequences = cgm_features_aligned.reshape(cgm_features_aligned.shape[0], 
                                                   cgm_features_aligned.shape[1], 1)
        
        print(f"CGM sequences for LSTM: {cgm_sequences.shape}")
        
        # Filter out samples with very small corrections (noise)
        correction_magnitude = np.mean(np.abs(correction_factors), axis=1)
        significant_mask = correction_magnitude > 1.0  # At least 1 mg/dL average correction
        
        cgm_filtered = cgm_sequences[significant_mask]
        corrections_filtered = correction_factors[significant_mask]
        
        print(f"Filtered to significant corrections: {len(cgm_filtered):,} samples")
        print(f"Average correction magnitude: {np.mean(np.abs(corrections_filtered)):.2f} mg/dL")
        
        return cgm_filtered, corrections_filtered
    
    def train_gt(self, simglucose_data, ft_system,
                       epochs: int = 150, batch_size: int = 64,
                       validation_split: float = 0.15,
                       early_stopping_patience: int = 40,
                       verbose: bool = True):
        """
        Train GT predictor
        """
        
        print("=" * 60)
        print("GT PREDICTOR TRAINING")
        print("=" * 60)
        print("Methodology: Learn insulin effects indirectly from CGM patterns")
        print("Input: CGM history only (past glucose trends)")
        print("Target: Correction factors for FT predictions")
        
        # Prepare training data
        cgm_sequences, correction_factors = self.prepare_gt_data(simglucose_data, ft_system)
        
        # Temporal split (forward chaining)
        split_idx = int(len(cgm_sequences) * (1 - validation_split))
        cgm_train = cgm_sequences[:split_idx]
        corrections_train = correction_factors[:split_idx]
        cgm_val = cgm_sequences[split_idx:]
        corrections_val = correction_factors[split_idx:]
        
        print(f"\\nTraining: {len(cgm_train):,}, Validation: {len(cgm_val):,}")
        print(f"CGM sequence length: {cgm_train.shape[1]}")
        print(f"Device: {self.device}")
        
        # Fit scalers on training data only
        # Reshape for scaler (samples * sequence_length, features)
        cgm_train_flat = cgm_train.reshape(-1, 1)
        self.scaler_cgm.fit(cgm_train_flat)
        self.scaler_corrections.fit(corrections_train)
        
        # Scale data
        cgm_train_scaled = self.scaler_cgm.transform(cgm_train_flat).reshape(cgm_train.shape)
        corrections_train_scaled = self.scaler_corrections.transform(corrections_train)
        
        cgm_val_flat = cgm_val.reshape(-1, 1)
        cgm_val_scaled = self.scaler_cgm.transform(cgm_val_flat).reshape(cgm_val.shape)
        corrections_val_scaled = self.scaler_corrections.transform(corrections_val)
        
        # Create data loaders
        train_dataset = GTDataset(cgm_train_scaled, corrections_train_scaled)
        val_dataset = GTDataset(cgm_val_scaled, corrections_val_scaled)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\\nStarting GT training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_cgm, batch_corrections in train_loader:
                batch_cgm = batch_cgm.to(self.device)
                batch_corrections = batch_corrections.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_cgm)
                
                loss = self.criterion(outputs, batch_corrections)
                
                # Gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            train_loss /= train_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_cgm, batch_corrections in val_loader:
                    batch_cgm = batch_cgm.to(self.device)
                    batch_corrections = batch_corrections.to(self.device)
                    outputs = self.model(batch_cgm)
                    
                    loss = self.criterion(outputs, batch_corrections)
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            if verbose and (epoch + 1) % 25 == 0:
                print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                      f'Train: {train_loss:.4f} | '
                      f'Val: {val_loss:.4f} | '
                      f'LR: {current_lr:.2e} | '
                      f'Best: {best_val_loss:.4f}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f'\\nEarly stopping at epoch {epoch+1}')
                    print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        
        # Save GT model
        model_path = 'gt_predictor.pth'
        torch.save(self.model.state_dict(), model_path)
        
        print(f"\\nGT training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved as '{model_path}'")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': best_val_loss
        }
    
    def predict_corrections(self, cgm_sequences: np.ndarray) -> np.ndarray:
        """Predict correction factors for FT predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Ensure proper shape (samples, sequence_length, 1)
        if len(cgm_sequences.shape) == 2:
            cgm_sequences = cgm_sequences.reshape(cgm_sequences.shape[0], 
                                                cgm_sequences.shape[1], 1)
        
        # Scale CGM input
        cgm_flat = cgm_sequences.reshape(-1, 1)
        cgm_scaled = self.scaler_cgm.transform(cgm_flat).reshape(cgm_sequences.shape)
        cgm_tensor = torch.FloatTensor(cgm_scaled).to(self.device)
        
        with torch.no_grad():
            corrections_scaled = self.model(cgm_tensor).cpu().numpy()
        
        # Inverse scale corrections
        corrections = self.scaler_corrections.inverse_transform(corrections_scaled)
        
        return corrections

def main():
    """Main GT training function"""
    
    print("GT PREDICTOR TRAINING")
    print("=" * 60)
    print("Input: CGM history only (glucose patterns)")
    print("Target: Correction factors for FT predictions")
    print("Learning: Insulin effects indirectly from CGM trends")
    print("Equation: ΔYcgm,FT = GT · ΔU (implicit)")
    
    # Load SimGlucose data
    try:
        with open('cache/temporal_simglucose_data.pkl', 'rb') as f:
            simglucose_data = pickle.load(f)
        
        print(f"Loaded SimGlucose temporal data")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load trained FT system
    try:
        from ft_trainer import FTSystem
        
        # Initialize FT system to match saved model
        ft_system = FTSystem(
            input_size=75,
            hidden_size=128,
            num_layers=2,
            prediction_horizon=8
        )
        
        # Load FT model and scalers - need to recreate the scalers
        # Prepare patient data the same way as in training
        all_X = np.vstack([simglucose_data['training']['scenario_1']['X'], 
                          simglucose_data['validation']['scenario_1']['X']])
        all_y = np.vstack([simglucose_data['training']['scenario_1']['y'], 
                          simglucose_data['validation']['scenario_1']['y']])
        all_patient_ids = np.hstack([simglucose_data['training']['scenario_1']['patient_ids'],
                                   simglucose_data['validation']['scenario_1']['patient_ids']])
        
        # Organize by patient (matching training approach)
        patient_data = {}
        unique_patients = sorted(np.unique(all_patient_ids))
        
        for patient_id in unique_patients:
            mask = all_patient_ids == patient_id
            patient_data[patient_id] = {
                'X': all_X[mask],
                'y': all_y[mask]
            }
        
        # Recreate training data for scaler fitting (using patients 0-8 like in final training iteration)
        train_patients = unique_patients[:-1]  # All but last patient
        X_train_list = []
        y_train_list = []
        for patient_id in train_patients:
            X_train_list.append(patient_data[patient_id]['X'])
            y_train_list.append(patient_data[patient_id]['y'])
        
        X_train_combined = np.vstack(X_train_list)
        y_train_combined = np.vstack(y_train_list)
        
        # Fit scalers on the same data used in final FT training iteration
        ft_system.scaler_X.fit(X_train_combined)
        ft_system.scaler_y.fit(y_train_combined)
        
        # Load the trained model weights
        ft_state = torch.load('ft_predictor.pth', map_location='cpu')
        ft_system.model.load_state_dict(ft_state)
        ft_system.model.eval()
        ft_system.is_fitted = True
        
        print(f"Loaded FT predictor")
        print(f"Recreated scalers for {len(train_patients)} training patients")
        
    except Exception as e:
        print(f"Error loading FT system: {e}")
        return
    
    # Initialize GT system
    gt = GTPredictor(
        cgm_history_length=25,  # CGM history length
        hidden_size=64,         # Smaller than FT predictor
        num_layers=2,
        prediction_horizon=8,
        learning_rate=0.001
    )
    
    # Train GT predictor
    history = gt.train_gt(
        simglucose_data, ft_system,
        epochs=100,
        batch_size=64,
        validation_split=0.15,
        early_stopping_patience=30,
        verbose=True
    )
    
    print("\\n" + "=" * 60)
    print("GT TRAINING COMPLETED")
    print("=" * 60)
    print("Key Features:")
    print("Learns insulin effects from CGM patterns only")
    print("Provides correction factors for FT predictions")
    print("Methodology: ΔYcgm,FT = GT · ΔU")
    print("Ready for LSTM-MPC integration")
    print("=" * 60)

if __name__ == "__main__":
    main()