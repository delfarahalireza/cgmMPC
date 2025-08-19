#!/usr/bin/env python3
"""
FT+GT Predictor
- FT predictor: Complete state vector → 8-step glucose predictions
- GT predictor: CGM patterns only → Correction factors for insulin effects
- Combined: Enhanced predictions accounting for insulin interventions
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import pickle
import warnings
from ft_trainer import FTSystem
from gt_trainer import GTSystem
warnings.filterwarnings('ignore')

class CombinedPredictor:
    """
    Combined FT+GT predictor system
    """
    
    def __init__(self):
        
        # Initialize FT system
        self.ft_system = FTSystem(
            input_size=75,
            hidden_size=128,
            num_layers=2,
            prediction_horizon=8
        )
        
        # Initialize GT system  
        self.gt_system = GTSystem(
            cgm_history_length=25,
            hidden_size=64,
            num_layers=2,
            prediction_horizon=8
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_loaded = False
        
    def load_trained_models(self, simglucose_data):
        """Load both trained FT and GT models"""
        
        
        try:
            # Load FT system
            print("Loading FT predictor...")
            
            # Recreate FT scalers (matching training approach)
            all_X = np.vstack([simglucose_data['training']['scenario_1']['X'], 
                              simglucose_data['validation']['scenario_1']['X']])
            all_y = np.vstack([simglucose_data['training']['scenario_1']['y'], 
                              simglucose_data['validation']['scenario_1']['y']])
            all_patient_ids = np.hstack([simglucose_data['training']['scenario_1']['patient_ids'],
                                       simglucose_data['validation']['scenario_1']['patient_ids']])
            
            # Organize by patient (matching FT training approach)
            patient_data = {}
            unique_patients = sorted(np.unique(all_patient_ids))
            
            for patient_id in unique_patients:
                mask = all_patient_ids == patient_id
                patient_data[patient_id] = {
                    'X': all_X[mask],
                    'y': all_y[mask]
                }
            
            # Use training data from final iteration (patients 0-8)
            train_patients = unique_patients[:-1]  # All but last patient
            X_train_list = []
            y_train_list = []
            for patient_id in train_patients:
                X_train_list.append(patient_data[patient_id]['X'])
                y_train_list.append(patient_data[patient_id]['y'])
            
            X_train_combined = np.vstack(X_train_list)
            y_train_combined = np.vstack(y_train_list)
            
            # Fit FT scalers
            self.ft_system.scaler_X.fit(X_train_combined)
            self.ft_system.scaler_y.fit(y_train_combined)
            
            # Load FT model weights
            ft_state = torch.load('ft_predictor.pth', map_location='cpu')
            self.ft_system.model.load_state_dict(ft_state)
            self.ft_system.model.eval()
            self.ft_system.is_fitted = True
            
            print("FT predictor loaded successfully")
            
            # Load GT system
            print("Loading GT predictor...")
            
            # Recreate GT data and scalers
            cgm_sequences, correction_factors = self.gt_system.prepare_gt_data(
                simglucose_data, self.ft_system
            )
            
            # Fit GT scalers on same data used in training
            split_idx = int(len(cgm_sequences) * 0.85)  # Same split as training
            cgm_train = cgm_sequences[:split_idx]
            corrections_train = correction_factors[:split_idx]
            
            cgm_train_flat = cgm_train.reshape(-1, 1)
            self.gt_system.scaler_cgm.fit(cgm_train_flat)
            self.gt_system.scaler_corrections.fit(corrections_train)
            
            # Load GT model weights
            gt_state = torch.load('gt_predictor.pth', map_location='cpu')
            self.gt_system.model.load_state_dict(gt_state)
            self.gt_system.model.eval()
            self.gt_system.is_fitted = True
            
            print("GT predictor loaded successfully")
            
            self.is_loaded = True
            
            print("\nCOMBINED SYSTEM READY")
            print(f"  FT System: Complete state vector → 8-step predictions")
            print(f"  GT System: CGM patterns → Correction factors")
            print(f"  Combined: Enhanced insulin-aware predictions")
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            raise
            
    def predict_ft_only(self, X: np.ndarray) -> np.ndarray:
        """FT predictions only (baseline)"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded first")
            
        return self.ft_system.predict(X)
    
    def predict_gt_corrections(self, X: np.ndarray) -> np.ndarray:
        """GT correction factors only"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded first")
            
        # Extract CGM features (first 25 features)
        cgm_features = X[:, :25]
        return self.gt_system.predict_corrections(cgm_features)
    
    def predict_combined(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Combined FT+GT predictions"""
        if not self.is_loaded:
            raise ValueError("Models must be loaded first")
            
        # Get FT predictions
        ft_predictions = self.predict_ft_only(X)
        
        # Get GT corrections
        gt_corrections = self.predict_gt_corrections(X)
        
        # Combine: FT + GT corrections
        combined_predictions = ft_predictions + gt_corrections
        
        return {
            'ft_only': ft_predictions,
            'gt_corrections': gt_corrections,
            'combined': combined_predictions
        }
    
    def evaluate_combined_system(self, test_X: np.ndarray, test_y: np.ndarray, 
                                scenario_name: str = "Test") -> Dict:
        """Comprehensive evaluation of combined system"""
        
        print(f"\nEVALUATING COMBINED SYSTEM - {scenario_name}")
        print("-" * 50)
        
        # Get predictions
        results = self.predict_combined(test_X)
        ft_pred = results['ft_only']
        gt_corr = results['gt_corrections']
        combined_pred = results['combined']
        
        # Calculate metrics
        def calc_metrics(predictions, ground_truth):
            mae = np.mean(np.abs(predictions - ground_truth))
            rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
            
            # Step-wise metrics
            step_mae = [np.mean(np.abs(predictions[:, i] - ground_truth[:, i])) 
                       for i in range(predictions.shape[1])]
            
            return {
                'mae': mae,
                'rmse': rmse,
                'step_mae': step_mae,
                'mae_15min': step_mae[0],
                'mae_2hour': step_mae[7]
            }
        
        ft_metrics = calc_metrics(ft_pred, test_y)
        combined_metrics = calc_metrics(combined_pred, test_y)
        
        # Calculate improvement
        mae_improvement = ((ft_metrics['mae'] - combined_metrics['mae']) / 
                          ft_metrics['mae'] * 100)
        
        print(f"Performance Comparison:")
        print(f"  FT Only MAE:    {ft_metrics['mae']:.2f} mg/dL")
        print(f"  Combined MAE:   {combined_metrics['mae']:.2f} mg/dL")
        print(f"  Improvement:    {mae_improvement:.1f}%")
        print(f"  GT Correction:  {np.mean(np.abs(gt_corr)):.2f} mg/dL (avg magnitude)")
        
        print(f"\nStep-wise Performance:")
        print("Step | Time   | FT MAE | Combined MAE | Improvement")
        print("-" * 55)
        for i in range(8):
            time_ahead = (i + 1) * 15
            ft_mae = ft_metrics['step_mae'][i]
            combined_mae = combined_metrics['step_mae'][i]
            improvement = (ft_mae - combined_mae) / ft_mae * 100
            print(f"{i+1:4d} | {time_ahead:3d}min | {ft_mae:6.2f} | {combined_mae:11.2f} | {improvement:9.1f}%")
        
        return {
            'ft_metrics': ft_metrics,
            'combined_metrics': combined_metrics,
            'improvement_percent': mae_improvement,
            'gt_correction_magnitude': np.mean(np.abs(gt_corr)),
            'predictions': results,
            'ground_truth': test_y
        }
    
    def plot_combined_predictions(self, evaluation_results: Dict, scenario_name: str = "Test"):
        """Plot combined system predictions"""
        
        predictions = evaluation_results['predictions']
        ground_truth = evaluation_results['ground_truth']
        
        ft_pred = predictions['ft_only']
        gt_corr = predictions['gt_corrections']
        combined_pred = predictions['combined']
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Plot 8 steps
        for step in range(8):
            row = step // 4
            col = step % 4
            ax = axes[row, col]
            
            time_ahead = (step + 1) * 15
            
            # Scatter plots
            ax.scatter(ground_truth[:, step], ft_pred[:, step], 
                      alpha=0.5, s=20, color='blue', label='FT Only')
            ax.scatter(ground_truth[:, step], combined_pred[:, step], 
                      alpha=0.5, s=20, color='red', label='FT+GT Combined')
            
            # Perfect prediction line
            min_val = min(np.min(ground_truth[:, step]), 
                         np.min(combined_pred[:, step]))
            max_val = max(np.max(ground_truth[:, step]), 
                         np.max(combined_pred[:, step]))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
            
            # Calculate MAEs
            ft_mae = np.mean(np.abs(ft_pred[:, step] - ground_truth[:, step]))
            combined_mae = np.mean(np.abs(combined_pred[:, step] - ground_truth[:, step]))
            improvement = (ft_mae - combined_mae) / ft_mae * 100
            
            ax.set_xlabel('Ground Truth (mg/dL)')
            ax.set_ylabel('Predictions (mg/dL)')
            ax.set_title(f'Step {step+1} ({time_ahead}min)\n'
                        f'FT: {ft_mae:.2f}, Combined: {combined_mae:.2f} mg/dL\n'
                        f'Improvement: {improvement:.1f}%')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.suptitle(f'FT+GT Prediction Performance - {scenario_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'ft_gt_performance_{scenario_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        # Plot correction factors
        self._plot_correction_analysis(gt_corr, scenario_name)
    
    def plot_24hour_population_average(self, simglucose_data: Dict, scenario_name: str = "Combined_System"):
        """Plot 24-hour population average ground truth vs combined predictions"""
        
        if not self.is_loaded:
            print("Models must be loaded first")
            return
            
        print(f"\nCreating 24-hour population average plot for combined system...")
        
        # Get all available data
        all_X = np.vstack([simglucose_data['training']['scenario_1']['X'], 
                          simglucose_data['validation']['scenario_1']['X']])
        all_y_s1 = np.vstack([simglucose_data['training']['scenario_1']['y'], 
                             simglucose_data['validation']['scenario_1']['y']])
        all_y_s2 = np.vstack([simglucose_data['training']['scenario_2']['y'], 
                             simglucose_data['validation']['scenario_2']['y']])
        all_patient_ids = np.hstack([simglucose_data['training']['scenario_1']['patient_ids'],
                                   simglucose_data['validation']['scenario_1']['patient_ids']])
        all_timestamps = np.hstack([simglucose_data['training']['scenario_1']['timestamps'],
                                   simglucose_data['validation']['scenario_1']['timestamps']])
        
        # Extract time features from the data
        time_features = all_X[:, -1]  # Last column is time of day
        
        # Get combined predictions for all data
        combined_results = self.predict_combined(all_X)
        all_ft_predictions = combined_results['ft_only']
        all_gt_corrections = combined_results['gt_corrections'] 
        all_combined_predictions = combined_results['combined']
        
        print(f"  Generated predictions for {len(all_X)} samples")
        
        # Create time bins (30-minute intervals for smoother curves)
        time_bins = np.arange(0, 24, 0.5)
        time_centers = time_bins[:-1] + 0.25
        
        # Calculate population averages for each time bin and each prediction step
        n_steps = 8
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Define colors for different prediction horizons
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
        
        # Plot ground truth (using scenario-2 as target with insulin effects)
        gt_profile_s1 = []
        gt_profile_s2 = []
        
        for i in range(len(time_bins) - 1):
            bin_mask = (time_features >= time_bins[i]) & (time_features < time_bins[i + 1])
            if np.any(bin_mask):
                # Scenario-1 ground truth (basal only)
                gt_avg_s1 = np.mean(all_y_s1[bin_mask, 0])  # Current glucose
                gt_profile_s1.append(gt_avg_s1)
                
                # Scenario-2 ground truth (basal + bolus)
                gt_avg_s2 = np.mean(all_y_s2[bin_mask, 0])  # Current glucose
                gt_profile_s2.append(gt_avg_s2)
            else:
                gt_profile_s1.append(np.nan)
                gt_profile_s2.append(np.nan)
        
        # Plot ground truth scenarios
        ax.plot(time_centers, gt_profile_s2, 'k', linewidth=3, label='Ground Truth', alpha=0.7)
        
        # Plot FT-only predictions (should match S1)
        ft_profile = []
        for i in range(len(time_bins) - 1):
            bin_mask = (time_features >= time_bins[i]) & (time_features < time_bins[i + 1])
            if np.any(bin_mask):
                ft_avg = np.mean(all_ft_predictions[bin_mask, 0])  # Current step prediction
                ft_profile.append(ft_avg)
            else:
                ft_profile.append(np.nan)

        
        # Plot combined predictions for selected steps
        selected_steps = [0, 1, 3, 7]  # 15min, 30min, 1h, 2h
        step_labels = ['15 min', '30 min', '1 hour', '2 hours']
        
        for i, step in enumerate(selected_steps):
            combined_profile = []
            for j in range(len(time_bins) - 1):
                bin_mask = (time_features >= time_bins[j]) & (time_features < time_bins[j + 1])
                if np.any(bin_mask):
                    combined_avg = np.mean(all_combined_predictions[bin_mask, step])
                    combined_profile.append(combined_avg)
                else:
                    combined_profile.append(np.nan)
            
            ax.plot(time_centers, combined_profile, '--', linewidth=2, 
                   color=colors[i], label=f'Multi-Step {step_labels[i]} ahead', alpha=0.8)
        
        # Add meal time indicators
        meal_times = [8, 13, 19]
        for meal_time in meal_times:
            ax.axvline(x=meal_time, color='red', linestyle=':', alpha=0.6, linewidth=2)
        
        # Formatting
        ax.set_xlabel('Time of Day (hours)', fontsize=12)
        ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        ax.set_title('24-Hour Population Average: Multi-Step Prediction vs Ground Truth', fontsize=14, fontweight='bold')
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
        filename = f'glucose_predictions_{scenario_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        # Print performance statistics
        print(f"\n24-Hour Population Average Performance:")
        
        # Calculate MAE between combined predictions and ground truth scenarios
        for i, step in enumerate(selected_steps):
            # MAE vs Scenario-1 (basal only)
            step_mae_s1 = np.mean(np.abs(all_combined_predictions[:, step] - all_y_s1[:, step]))
            # MAE vs Scenario-2 (basal + bolus) 
            step_mae_s2 = np.mean(np.abs(all_combined_predictions[:, step] - all_y_s2[:, step]))
            
            print(f"  {step_labels[i]} ahead:")
            print(f"    vs S1 (basal): {step_mae_s1:.2f} mg/dL")
            print(f"    vs S2 (bolus): {step_mae_s2:.2f} mg/dL")
        
        # Overall insulin effect analysis
        avg_insulin_effect_gt = np.mean(all_y_s2 - all_y_s1)
        avg_gt_corrections = np.mean(all_gt_corrections)
        
        print(f"\nInsulin Effect Analysis:")
        print(f"  Ground truth insulin effect: {avg_insulin_effect_gt:.2f} mg/dL")
        print(f"  GT predicted corrections: {avg_gt_corrections:.2f} mg/dL") 
        print(f"  GT accuracy: {(1 - abs(avg_gt_corrections - avg_insulin_effect_gt) / abs(avg_insulin_effect_gt)) * 100:.1f}%")
    
    def _plot_correction_analysis(self, corrections: np.ndarray, scenario_name: str):
        """Plot GT correction factor analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Correction magnitude by step
        ax1 = axes[0, 0]
        step_corrections = [np.mean(np.abs(corrections[:, i])) for i in range(8)]
        time_steps = [(i+1)*15 for i in range(8)]
        
        ax1.bar(range(8), step_corrections, color='purple', alpha=0.7)
        ax1.set_xlabel('Prediction Step')
        ax1.set_ylabel('Average |Correction| (mg/dL)')
        ax1.set_title('GT Correction Magnitude by Step')
        ax1.set_xticks(range(8))
        ax1.set_xticklabels([f'{t}min' for t in time_steps], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correction distribution
        ax2 = axes[0, 1]
        all_corrections = corrections.flatten()
        ax2.hist(all_corrections, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No correction')
        ax2.axvline(x=np.mean(all_corrections), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_corrections):.2f}')
        ax2.set_xlabel('Correction Factor (mg/dL)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of GT Corrections')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Step-wise correction patterns
        ax3 = axes[1, 0]
        for step in range(0, 8, 2):  # Show every other step for clarity
            ax3.plot(corrections[:200, step], alpha=0.6, 
                    label=f'Step {step+1} ({(step+1)*15}min)')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Correction Factor (mg/dL)')
        ax3.set_title('GT Correction Patterns (First 200 samples)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correction statistics
        ax4 = axes[1, 1]
        correction_stats = {
            'Mean': np.mean(all_corrections),
            'Std': np.std(all_corrections),
            'Min': np.min(all_corrections),
            'Max': np.max(all_corrections),
            'Median': np.median(all_corrections)
        }
        
        bars = ax4.bar(correction_stats.keys(), correction_stats.values(), 
                      color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
        ax4.set_ylabel('Value (mg/dL)')
        ax4.set_title('GT Correction Statistics')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, correction_stats.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle(f'GT Correction Analysis - {scenario_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'gt_correction_analysis_{scenario_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main combined system testing function"""
    
    print("FT+GT SYSTEM TESTING")
    print("=" * 60)
    
    # Load SimGlucose data
    try:
        with open('cache/temporal_simglucose_data.pkl', 'rb') as f:
            simglucose_data = pickle.load(f)
        print("Loaded SimGlucose temporal data")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Initialize and load combined system
    combined_system = CombinedPredictor()
    combined_system.load_trained_models(simglucose_data)
    
    # Test on validation data
    val_X = simglucose_data['validation']['scenario_1']['X']
    val_y_s2 = simglucose_data['validation']['scenario_2']['y']  # Test insulin effects
    
    print(f"\nTest data: {len(val_X):,} samples")
    
    # Evaluate combined system on insulin scenario
    results = combined_system.evaluate_combined_system(
        val_X, val_y_s2, "Insulin_Scenario"
    )
    
    # Create visualization
    combined_system.plot_combined_predictions(results, "Insulin_Scenario")
    combined_system.plot_24hour_population_average(simglucose_data)

    
    print("\n" + "=" * 60)
    print("COMBINED SYSTEM TESTING COMPLETED!")
    print("=" * 60)
    print(f"FT+GT integration successful")
    print(f"Performance improvement: {results['improvement_percent']:.1f}%")
    print(f"GT provides meaningful corrections: {results['gt_correction_magnitude']:.2f} mg/dL")
    print("Ready for Brown dataset validation")
    print("=" * 60)

if __name__ == "__main__":
    main()