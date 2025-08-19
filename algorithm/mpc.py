#!/usr/bin/env python3
"""
LSTM-MPC Implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
from datetime import datetime, timedelta
from combined_predictor import CombinedPredictor
warnings.filterwarnings('ignore')

class MPC:
    """
    Calibrated LSTM-MPC with safe GT scaling
    """
    
    def __init__(self, sampling_minutes: int = 15, gt_scaling_factor: float = 0.5):
        self.sampling_minutes = sampling_minutes
        
        # MPC parameters (from paper)
        self.T = 8              # Prediction horizon (120 minutes)
        self.q = 1.0            # Glucose tracking weight
        self.r = 25.0           # Insulin deviation weight (higher = more selective)
        
        # Calibrated GT scaling
        self.gt_scaling_factor = gt_scaling_factor
        
        # Time-dependent setpoints (from paper)
        self.setpoint_day = 110.0    # mg/dL (5:00 AM - 10:00 PM)
        self.setpoint_night = 125.0  # mg/dL (10:00 PM - 5:00 AM)
        
        # Hardware constraints (from paper)
        self.Y_min = 70.0       # mg/dL (safer minimum)
        self.Y_max = 300.0      # mg/dL (safer maximum)
        self.U_min = 0.0        # U min
        self.U_max = 3.0        # U (much safer maximum)
        
        # Subject-dependent basal insulin rate
        self.basal_rate = 1.0   # U/hr
        
        # Load trained FT+GT system
        self.predictor = CombinedPredictor()
        
        print(f"  GT scaling factor: {gt_scaling_factor:.3f} (calibrated)")
        print(f"  Prediction horizon T: {self.T} steps ({self.T * sampling_minutes} min)")
        print(f"  Safety constraints: Y∈[{self.Y_min},{self.Y_max}], U∈[{self.U_min},{self.U_max}]")
        print(f"  MPC weights: q={self.q}, r={self.r}")
    
    def load_predictors(self, data_path: str = 'cache/temporal_simglucose_data.pkl'):
        """Load trained FT+GT prediction models"""
        
        print("\nLoading FT+GT predictors...")
        
        try:
            with open(data_path, 'rb') as f:
                simglucose_data = pickle.load(f)
            
            self.predictor.load_trained_models(simglucose_data)
            print("FT+GT predictors loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading predictors: {e}")
            return False
    
    def get_setpoint_vector(self, current_time: datetime) -> np.ndarray:
        """Get time-dependent setpoint vector Y̅T"""
        
        setpoints = []
        
        for step in range(self.T):
            step_time = current_time + timedelta(minutes=step * self.sampling_minutes)
            hour = step_time.hour
            
            # setpoint logic
            if 5 <= hour < 22:
                setpoints.append(self.setpoint_day)   # 110 mg/dL
            else:
                setpoints.append(self.setpoint_night) # 125 mg/dL
        
        return np.array(setpoints)
    
    def get_basal_vector(self) -> np.ndarray:
        """Get basal insulin vector ŪT-1"""
        
        basal_per_step = self.basal_rate * (self.sampling_minutes / 60.0)
        return np.full(self.T, basal_per_step)
    
    def predict_glucose_calibrated(self, x_k: np.ndarray, U_T: np.ndarray) -> np.ndarray:
        """
        Glucose prediction:
        ŶT(xk) = FT(xk) + GT_scaled(xk) * (UT-1 - ŪT-1)
        
        Where GT_scaled uses the calibrated scaling factor
        """
        
        # Get FT and GT predictions
        predictions = self.predictor.predict_combined(x_k.reshape(1, -1))
        
        # FT(xk): Base glucose predictions
        FT_pred = predictions['ft_only'][0]  # (8,)
        
        # GT corrections (already computed)
        GT_corrections = predictions['gt_corrections'][0]  # (8,)
        
        # Get basal vector and insulin deviation
        U_basal = self.get_basal_vector()  # (8,)
        U_deviation = U_T - U_basal  # (8,)
        
        # Model: ŶT = FT(xk) + GT_scaled * (UT-1 - ŪT-1)
        # Use calibrated scaling factor
        GT_per_unit_scaled = GT_corrections * self.gt_scaling_factor
        
        # Apply to insulin deviation (step-wise scaling)
        # GT corrections should be NEGATIVE when insulin is INCREASED (more insulin = lower glucose)
        GT_effect = GT_per_unit_scaled * U_deviation
        
        # Final prediction
        Y_hat = FT_pred + GT_effect
        
        # Debug insulin direction around meals vs fasting
        if hasattr(self, 'current_carbs') and (self.current_carbs > 0 or np.any(U_deviation > 0.5)):
            hour = getattr(self, 'current_hour', 0)
            meal_flag = "MEAL" if getattr(self, 'current_carbs', 0) > 0 else "FASTING"
            print(f"  DEBUG {meal_flag} at {hour:.1f}h:")
            print(f"    U_deviation: {U_deviation[0]:+.3f} U")
            print(f"    GT_raw: {GT_corrections[0]:.2f} mg/dL")
            print(f"    GT_effect: {GT_effect[0]:+.2f} mg/dL")
            print(f"    Prediction: {FT_pred[0]:.1f} → {Y_hat[0]:.1f} mg/dL")
        
        return Y_hat
    
    def cost_funtion(self, U_T: np.ndarray, x_k: np.ndarray, 
                                      Y_setpoint: np.ndarray, U_basal: np.ndarray, 
                                      current_time: Optional[datetime] = None) -> float:
        """MPC cost function"""
        
        # Predict glucose trajectory with calibrated GT
        Y_hat = self.predict_glucose_calibrated(x_k, U_T)  # (8,)
        
        # Glucose tracking cost
        glucose_error = Y_hat - Y_setpoint  # (8,)
        glucose_cost = self.q * np.sum(glucose_error ** 2)
        
        # Meal-aware insulin deviation cost
        insulin_deviation = U_T - U_basal  # (8,)
        
        # Adjust insulin penalty based on meal proximity
        if current_time is not None:
            meal_penalty_factor = self._get_meal_penalty_factor(current_time)
        else:
            meal_penalty_factor = 1.0
            
        insulin_cost = self.r * meal_penalty_factor * np.sum(insulin_deviation ** 2)
        
        total_cost = glucose_cost + insulin_cost
        
        return total_cost
    
    def _get_meal_penalty_factor(self, current_time: datetime) -> float:
        """Get meal-aware penalty factor (lower around meals = more aggressive insulin)"""
        
        hour = current_time.hour + current_time.minute / 60.0
        
        # Define meal windows (hour ± 1.5 hours around each meal)
        meal_times = [7.5, 12.5, 18.5, 21.0]  # Breakfast, Lunch, Dinner, Snack
        meal_window = 1.5  # Hours before/after meal
        
        # Check if we're in any meal window
        in_meal_window = False
        for meal_time in meal_times:
            if abs(hour - meal_time) <= meal_window:
                in_meal_window = True
                break
        
        # Dawn period (4-8 AM) - be more conservative due to dawn phenomenon
        if 4.0 <= hour <= 8.0:
            if in_meal_window:
                return 0.4  # Moderately aggressive around breakfast during dawn
            else:
                return 2.5  # More conservative during dawn fasting (higher penalty)
        
        # Regular periods
        if in_meal_window:
            return 0.3  # Low penalty around meals (more aggressive insulin)
        else:
            return 2.0  # High penalty during fasting (conservative insulin)
    
    def constraints_calibrated(self, U_T: np.ndarray, x_k: np.ndarray) -> List[Dict]:
        """Calibrated safety constraints"""
        
        constraints = []
        
        # Glucose constraints (stricter for safety)
        def glucose_min_constraint(U):
            Y_hat = self.predict_glucose_calibrated(x_k, U)
            return np.min(Y_hat) - self.Y_min  # Y_hat >= 70 mg/dL
        
        def glucose_max_constraint(U):
            Y_hat = self.predict_glucose_calibrated(x_k, U)
            return self.Y_max - np.max(Y_hat)  # Y_hat <= 300 mg/dL
        
        constraints.extend([
            {'type': 'ineq', 'fun': glucose_min_constraint},
            {'type': 'ineq', 'fun': glucose_max_constraint}
        ])
        
        # Additional safety constraint: limit total insulin per hour
        def total_insulin_constraint(U):
            total_per_hour = np.sum(U) * (60 / self.sampling_minutes)  # Convert to U/hr
            return 10.0 - total_per_hour  # Max 10 U/hr total
        
        constraints.append({'type': 'ineq', 'fun': total_insulin_constraint})
        
        return constraints
    
    def solve_calibrated_mpc(self, x_k: np.ndarray, current_time: datetime) -> Dict:
        """Solve MPC optimization"""
        
        print(f"\nCALIBRATED MPC OPTIMIZATION")
        print(f"  Time: {current_time.strftime('%H:%M')}")
        
        # Get setpoints and basal
        Y_setpoint = self.get_setpoint_vector(current_time)
        U_basal = self.get_basal_vector()
        meal_penalty = self._get_meal_penalty_factor(current_time)
        
        print(f"  Setpoint: {Y_setpoint[0]:.0f} mg/dL")
        print(f"  Basal: {U_basal[0]:.3f} U/step")
        print(f"  Meal penalty factor: {meal_penalty:.1f} ({'MEAL WINDOW' if meal_penalty < 1.0 else 'FASTING'})")
        
        # Initial guess: basal insulin
        U_initial = U_basal.copy()
        
        # Bounds: stricter for safety
        bounds = [(self.U_min, self.U_max) for _ in range(self.T)]
        
        # Constraints
        constraints = self.constraints_calibrated(U_initial, x_k)
        
        try:
            # Solve optimization
            result = minimize(
                fun=self.cost_funtion,
                x0=U_initial,
                args=(x_k, Y_setpoint, U_basal, current_time),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success:
                U_optimal = result.x
                Y_predicted = self.predict_glucose_calibrated(x_k, U_optimal)
                
                # Calculate cost components
                glucose_error = Y_predicted - Y_setpoint
                glucose_cost = self.q * np.sum(glucose_error ** 2)
                insulin_deviation = U_optimal - U_basal
                insulin_cost = self.r * np.sum(insulin_deviation ** 2)
                
                # Receding horizon: take first control action
                u_mpc = U_optimal[0]
                
                print(f"  Optimization successful")
                print(f"  Control action: {u_mpc:.3f} U")
                print(f"  Insulin deviation: {u_mpc - U_basal[0]:.3f} U")
                print(f"  Cost: {result.fun:.1f} (Glucose: {glucose_cost:.1f}, Insulin: {insulin_cost:.1f})")
                print(f"  Predicted glucose: {np.min(Y_predicted):.1f}-{np.max(Y_predicted):.1f} mg/dL")
                
                return {
                    'success': True,
                    'u_mpc': u_mpc,
                    'U_optimal': U_optimal,
                    'Y_predicted': Y_predicted,
                    'Y_setpoint': Y_setpoint,
                    'total_cost': result.fun,
                    'glucose_cost': glucose_cost,
                    'insulin_cost': insulin_cost,
                    'insulin_deviation': u_mpc - U_basal[0]
                }
            
            else:
                print(f"Optimization failed: {result.message}")
                
                # Fallback: use basal insulin
                u_mpc = U_basal[0]
                Y_predicted = self.predict_glucose_calibrated(x_k, U_basal)
                
                return {
                    'success': False,
                    'u_mpc': u_mpc,
                    'Y_predicted': Y_predicted,
                    'Y_setpoint': Y_setpoint,
                    'fallback': 'basal',
                    'message': result.message
                }
                
        except Exception as e:
            print(f"  ✗ Optimization error: {e}")
            
            # Emergency fallback
            u_mpc = U_basal[0]  # Use basal rate
            
            return {
                'success': False,
                'u_mpc': u_mpc,
                'error': str(e),
                'fallback': 'emergency'
            }
    
    def create_state_vector(self, cgm_history: List[float], insulin_history: List[float],
                          carb_history: List[float], current_time: datetime) -> np.ndarray:
        """Create state vector matching training format"""
        
        def pad_history(history, target_length=25, default_value=0.0):
            if len(history) >= target_length:
                return history[-target_length:]
            else:
                padding = [default_value] * (target_length - len(history))
                return padding + history
        
        cgm_hist = pad_history(cgm_history, 25, 120.0)
        insulin_hist = pad_history(insulin_history, 25, 0.0)
        carb_hist = pad_history(carb_history, 25, 0.0)
        
        current_hour = current_time.hour + current_time.minute / 60.0
        
        state_vector = np.array(cgm_hist + insulin_hist + carb_hist + [current_hour])
        
        if len(state_vector) > 75:
            state_vector = state_vector[:75]
        elif len(state_vector) < 75:
            state_vector = np.pad(state_vector, (0, 75 - len(state_vector)), 'constant')
        
        return state_vector
    
    def simulate_calibrated_mpc(self, initial_cgm: float, simulation_hours: int = 8,
                              meal_plan: List[Tuple] = None, patient_basal_rate: float = 1.0) -> Dict:
        """Simulate MPC"""
        
        print(f"\nSIMULATING LSTM-MPC")
        print(f"  Duration: {simulation_hours} hours")
        print(f"  Patient basal rate: {patient_basal_rate:.2f} U/hr")
        print(f"  GT scaling factor: {self.gt_scaling_factor:.3f}")
        
        # Set patient-specific basal
        self.basal_rate = patient_basal_rate
        
        # Initialize simulation
        current_time = datetime(2024, 1, 1, 0, 0, 0)  # Start at midnight (12 AM)
        current_cgm = initial_cgm
        
        # Initialize histories
        cgm_history = [current_cgm] * 5
        insulin_history = [patient_basal_rate * 0.25] * 5  # 15-min basal doses
        carb_history = [0.0] * 5
        
        # Results storage
        results = {
            'times': [current_time],
            'glucose': [current_cgm],
            'insulin': [],
            'setpoints': [],
            'predictions': [],
            'costs': [],
            'insulin_deviations': [],
            'control_actions': []
        }
        
        n_steps = simulation_hours * 60 // self.sampling_minutes
        
        for step in range(n_steps):
            print(f"\nStep {step+1}/{n_steps} - {current_time.strftime('%H:%M')}")
            
            # Check for meals
            current_carbs = 0.0
            if meal_plan:
                for meal_time, meal_carbs in meal_plan:
                    if abs((current_time.hour + current_time.minute/60.0) - meal_time) < 0.1:
                        current_carbs = meal_carbs
                        print(f"Meal: {meal_carbs}g carbs")
            
            # Create state vector
            x_k = self.create_state_vector(cgm_history, insulin_history, carb_history, current_time)
            
            # Solve calibrated MPC
            mpc_result = self.solve_calibrated_mpc(x_k, current_time)
            
            # Extract control action
            u_mpc = mpc_result['u_mpc']
            
            # Store results
            results['insulin'].append(u_mpc)
            results['control_actions'].append(mpc_result)
            
            if 'Y_setpoint' in mpc_result:
                results['setpoints'].append(mpc_result['Y_setpoint'][0])
            if 'Y_predicted' in mpc_result:
                results['predictions'].append(mpc_result['Y_predicted'])
            if 'total_cost' in mpc_result:
                results['costs'].append(mpc_result['total_cost'])
            if 'insulin_deviation' in mpc_result:
                results['insulin_deviations'].append(mpc_result['insulin_deviation'])
            
            # Update histories
            cgm_history.append(current_cgm)
            insulin_history.append(u_mpc)
            carb_history.append(current_carbs)
            
            # Keep reasonable history length
            if len(cgm_history) > 25:
                cgm_history = cgm_history[-25:]
                insulin_history = insulin_history[-25:]
                carb_history = carb_history[-25:]
            
            # Simulate glucose evolution (physiological model)
            hour_of_day = current_time.hour + current_time.minute / 60.0
            
            # Dawn phenomenon (glucose rises 4-8 AM)
            if 4 <= hour_of_day <= 8:
                dawn_effect = 15 * np.sin((hour_of_day - 4) * np.pi / 4)  # Peak at 6 AM
            else:
                dawn_effect = 0
            
            # Circadian insulin sensitivity variation
            if 2 <= hour_of_day <= 10:  # Morning insulin resistance
                insulin_sensitivity = 0.7
            elif 10 <= hour_of_day <= 18:  # Day time normal
                insulin_sensitivity = 1.0
            else:  # Evening/night increased sensitivity
                insulin_sensitivity = 1.3
                
            # Natural glucose regulation (stronger)
            glucose_drift = -0.05 * (current_cgm - 120)
            
            # Insulin effect (improved)
            insulin_deviation = u_mpc - patient_basal_rate * 0.25
            insulin_effect = -insulin_deviation * 12.0 * insulin_sensitivity
            
            # Carbohydrate absorption (more realistic)
            if current_carbs > 0:
                carb_effect = current_carbs * 2.5  # Stronger carb effect
            else:
                carb_effect = 0
            
            # Physiological noise
            noise = np.random.normal(0, 4)
            
            # Combined effects
            total_change = glucose_drift + insulin_effect + carb_effect + dawn_effect + noise
            current_cgm = current_cgm + total_change
            current_cgm = np.clip(current_cgm, 60, 400)  # Physiological bounds
            
            # Update time
            current_time += timedelta(minutes=self.sampling_minutes)
            results['times'].append(current_time)
            results['glucose'].append(current_cgm)
        
        # Calculate summary metrics
        glucose_array = np.array(results['glucose'])
        tir_70_180 = np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100
        tir_70_250 = np.sum((glucose_array >= 70) & (glucose_array <= 250)) / len(glucose_array) * 100
        
        results['summary'] = {
            'time_in_range_70_180': tir_70_180,
            'time_in_range_70_250': tir_70_250,
            'mean_glucose': np.mean(glucose_array),
            'glucose_std': np.std(glucose_array),
            'total_insulin': np.sum(results['insulin']) if results['insulin'] else 0,
            'mean_insulin_deviation': np.mean(np.abs(results['insulin_deviations'])) if results['insulin_deviations'] else 0,
            'hypoglycemia_events': np.sum(glucose_array < 70),
            'severe_hypoglycemia_events': np.sum(glucose_array < 54),
            'hyperglycemia_events': np.sum(glucose_array > 250),
            'mean_cost': np.mean(results['costs']) if results['costs'] else 0,
            'glucose_cv': np.std(glucose_array) / np.mean(glucose_array) * 100
        }
        
        print(f"\n CALIBRATED MPC SUMMARY:")
        print(f"  Time in Range (70-180): {tir_70_180:.1f}%")
        print(f"  Time in Range (70-250): {tir_70_250:.1f}%")
        print(f"  Mean glucose: {results['summary']['mean_glucose']:.1f} ± {results['summary']['glucose_std']:.1f} mg/dL")
        print(f"  Glucose CV: {results['summary']['glucose_cv']:.1f}%")
        print(f"  Total insulin: {results['summary']['total_insulin']:.2f} U")
        print(f"  Mean insulin deviation: {results['summary']['mean_insulin_deviation']:.3f} U")
        print(f"  Hypoglycemia events: {results['summary']['hypoglycemia_events']}")
        print(f"  Severe hypoglycemia: {results['summary']['severe_hypoglycemia_events']}")
        
        return results
    
    def plot_24hour_mpc_control(self, results: Dict, scenario_name: str = "MPC_Control"):
        """Plot 24-hour MPC glucose control with population average style"""
        
        if not results or 'glucose' not in results:
            print("No results to plot")
            return
            
        print(f"\nCreating 24-hour MPC control plot...")
        
        # Convert times to hours from start
        times_hours = [(t - results['times'][0]).total_seconds() / 3600 for t in results['times']]
        glucose_values = np.array(results['glucose'])
        insulin_values = np.array(results['insulin']) if results['insulin'] else []
        setpoint_values = np.array(results['setpoints']) if results['setpoints'] else []
        
        # Create time bins for averaging (30-minute intervals) - always 24 hours
        time_bins = np.arange(0, 24.5, 0.5)
        time_centers = time_bins[:-1] + 0.25
        
        # Calculate averages and standard deviations for each time bin (simulate population average)
        glucose_profile = []
        glucose_std_profile = []
        insulin_profile = []
        
        for i in range(len(time_bins) - 1):
            # Find samples in this time bin
            bin_mask = (np.array(times_hours) >= time_bins[i]) & (np.array(times_hours) < time_bins[i + 1])
            
            if np.any(bin_mask):
                # Average glucose and std in this time bin
                glucose_avg = np.mean(glucose_values[bin_mask])
                glucose_std = np.std(glucose_values[bin_mask]) if np.sum(bin_mask) > 1 else 0.0
                glucose_profile.append(glucose_avg)
                glucose_std_profile.append(glucose_std)
                
                # Average insulin (offset by 1 since insulin starts from step 1)
                if len(insulin_values) > 0:
                    insulin_mask = bin_mask[1:len(insulin_values)+1] if len(bin_mask) > len(insulin_values) else bin_mask[:len(insulin_values)]
                    if np.any(insulin_mask):
                        insulin_avg = np.mean(insulin_values[insulin_mask])
                        insulin_profile.append(insulin_avg)
                    else:
                        insulin_profile.append(np.nan)
                else:
                    insulin_profile.append(np.nan)
            else:
                glucose_profile.append(np.nan)
                glucose_std_profile.append(0.0)
                insulin_profile.append(np.nan)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Glucose control with standard deviation
        # Convert to numpy arrays for easier manipulation
        glucose_profile = np.array(glucose_profile)
        glucose_std_profile = np.array(glucose_std_profile)
        
        # Plot main glucose curve
        ax1.plot(time_centers, glucose_profile, 'b-', linewidth=3, label='MPC Glucose Control', alpha=0.9)
        
        # Add standard deviation shading
        ax1.fill_between(time_centers, 
                        glucose_profile - glucose_std_profile,
                        glucose_profile + glucose_std_profile,
                        alpha=0.2, color='blue', label='± 1 SD')
        
        # Only target range background (70-180)
        ax1.axhspan(70, 180, alpha=0.15, color='green', label='Target Range (70-180)')
        
        # Meal time indicators
        meal_times = [7.5, 12.5, 18.5, 21.0]
        meal_names = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
        for meal_time, meal_name in zip(meal_times, meal_names):
            if meal_time <= max(time_centers):
                ax1.axvline(x=meal_time, color='brown', linestyle=':', alpha=0.8, linewidth=2)
                ax1.text(meal_time, 320, meal_name, rotation=90, ha='center', va='bottom', 
                        fontsize=10, color='brown', fontweight='bold')
        
        ax1.set_xlabel('Time of Day (hours)', fontsize=12)
        ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        ax1.set_title('MPC Glucose Control: Population Average', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25, 3))
        ax1.set_ylim(50, 350)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper right')
        
        # Add MPC performance summary in upper left corner
        glucose_array = np.array(results['glucose'])
        total_insulin = np.sum(results['insulin']) if results['insulin'] else 0
        tir_70_180 = np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100
        mean_glucose = np.mean(glucose_array)
        glucose_std = np.std(glucose_array)
        glucose_cv = glucose_std / mean_glucose * 100
        hypo_events = np.sum(glucose_array < 70)
        
        summary_text = f"""MPC Performance Summary:
Time in Range (70-180): {tir_70_180:.1f}%
Mean Glucose: {mean_glucose:.1f}±{glucose_std:.1f} mg/dL
Glucose CV: {glucose_cv:.1f}%
Total Insulin: {total_insulin:.1f} U
Hypoglycemic Events: {hypo_events}"""
        
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom plot: Insulin delivery
        ax2.plot(time_centers, insulin_profile, 'r-', linewidth=3, label='MPC Insulin Delivery', alpha=0.9)
        
        # Basal reference line
        basal_per_step = 1.2 * 0.25  # Convert 1.2 U/hr to 15-min doses
        ax2.axhline(y=basal_per_step, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Basal Rate')
        
        # Meal time indicators
        for meal_time, meal_name in zip(meal_times, meal_names):
            if meal_time <= max(time_centers):
                ax2.axvline(x=meal_time, color='brown', linestyle=':', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Time of Day (hours)', fontsize=12)
        ax2.set_ylabel('Insulin Delivery (U per 15min)', fontsize=12)
        ax2.set_title('MPC Insulin Delivery Profile', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 24)
        ax2.set_xticks(range(0, 25, 3))
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'mpc_control_{scenario_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        

        # Print detailed performance statistics
        print(f"\nMPC Control Performance:")
        
        # Calculate time-based statistics
        glucose_array = np.array(results['glucose'])
        
        # Time in Range analysis
        tir_70_180 = np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100
        tir_70_250 = np.sum((glucose_array >= 70) & (glucose_array <= 250)) / len(glucose_array) * 100
        
        # Glycemic episodes
        hypo_mild = np.sum((glucose_array >= 54) & (glucose_array < 70))
        hypo_severe = np.sum(glucose_array < 54)
        hyper_mild = np.sum((glucose_array > 180) & (glucose_array <= 250))
        hyper_severe = np.sum(glucose_array > 250)
        
        # Insulin statistics
        if results['insulin']:
            total_insulin = np.sum(results['insulin'])
            mean_insulin = np.mean(results['insulin'])
            basal_equivalent = basal_per_step * len(results['insulin'])
            insulin_deviation = total_insulin - basal_equivalent
        else:
            total_insulin = mean_insulin = basal_equivalent = insulin_deviation = 0
        
        print(f"  Glucose Control:")
        print(f"    Mean glucose: {np.mean(glucose_array):.1f} ± {np.std(glucose_array):.1f} mg/dL")
        print(f"    Time in Range (70-180): {tir_70_180:.1f}%")
        print(f"    Time in Range (70-250): {tir_70_250:.1f}%")
        print(f"    Coefficient of Variation: {(np.std(glucose_array)/np.mean(glucose_array)*100):.1f}%")
        
        print(f"  Glycemic Episodes:")
        print(f"    Mild hypoglycemia (54-70): {hypo_mild} events")
        print(f"    Severe hypoglycemia (<54): {hypo_severe} events") 
        print(f"    Mild hyperglycemia (180-250): {hyper_mild} events")
        print(f"    Severe hyperglycemia (>250): {hyper_severe} events")
        
        print(f"  Insulin Delivery:")
        print(f"    Total insulin: {total_insulin:.2f} U")
        print(f"    Mean per step: {mean_insulin:.3f} U")
        print(f"    Basal equivalent: {basal_equivalent:.2f} U")
        print(f"    Net insulin deviation: {insulin_deviation:+.2f} U")
        
        # Clinical assessment
        if hypo_severe == 0 and tir_70_180 > 70:
            assessment = "EXCELLENT - Safe and effective glucose control"
        elif hypo_severe == 0 and tir_70_180 > 50:
            assessment = "GOOD - Safe control with room for improvement"
        elif hypo_severe == 0:
            assessment = "ACCEPTABLE - Safe but sub-optimal control"
        else:
            assessment = "NEEDS IMPROVEMENT - Safety concerns detected"
        
        print(f"  Clinical Assessment: {assessment}")

def main():
    """Main calibrated MPC testing function"""
    
    print("LSTM-MPC IMPLEMENTATION")
    
    # Initialize calibrated MPC
    mpc = MPC(
        sampling_minutes=15,
        gt_scaling_factor=0.5  # Balanced scaling for safe active control
    )
    
    # Load predictors
    if not mpc.load_predictors():
        print("Failed to load predictors. Exiting.")
        return
    
    # Define realistic 24-hour meal plan (starting from midnight)
    meal_plan = [
        (7.5, 50),   # Breakfast: 7:30 AM
        (12.5, 65),  # Lunch: 12:30 PM  
        (18.5, 80),  # Dinner: 6:30 PM
        (21.0, 25)   # Evening snack: 9:00 PM
    ]
    
    # Simulate calibrated MPC for 24 hours
    results = mpc.simulate_calibrated_mpc(
        initial_cgm=150.0,
        simulation_hours=24,  # Full 24-hour simulation
        meal_plan=meal_plan,
        patient_basal_rate=1.2  # Standard basal rate
    )
    

    # Plot 24-hour population-average
    mpc.plot_24hour_mpc_control(results, "")
    
    print("\n" + "=" * 70)
    print("MPC SIMULATION COMPLETED!")
    print("=" * 70)

def plot_calibrated_results(results: Dict, meal_plan: List[Tuple] = None):
    """Plot calibrated MPC results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    times_hours = [(t - results['times'][0]).total_seconds() / 3600 for t in results['times']]
    glucose_times = times_hours
    insulin_times = times_hours[1:]

    # Plot 1: Glucose control
    ax1 = axes[0, 0]
    ax1.plot(glucose_times, results['glucose'], 'b-', linewidth=2, label='Glucose')

    if results['setpoints']:
        ax1.plot(insulin_times, results['setpoints'], 'g--', linewidth=2, alpha=0.7, label='Setpoint')

    # Safety zones
    ax1.axhspan(70, 180, alpha=0.2, color='green', label='Target Range')
    ax1.axhspan(54, 70, alpha=0.2, color='yellow', label='Mild Hypo')
    ax1.axhspan(0, 54, alpha=0.2, color='red', label='Severe Hypo')
    ax1.axhspan(250, 400, alpha=0.2, color='orange', label='Hyperglycemia')

    # Meal markers
    if meal_plan:
        for meal_time, meal_carbs in meal_plan:
            if meal_time <= max(glucose_times):
                ax1.axvline(x=meal_time, color='brown', linestyle=':', alpha=0.8)
                ax1.text(meal_time, 300, f'{meal_carbs}g', rotation=90, ha='center')

    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title('Calibrated MPC Glucose Control')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 350)

    # Plot 2: Insulin delivery
    ax2 = axes[0, 1]
    if results['insulin']:
        ax2.step(insulin_times, results['insulin'], 'r-', linewidth=2, where='post', label='MPC Insulin')

        basal_ref = [1.2 * 0.25] * len(results['insulin'])
        ax2.step(insulin_times, basal_ref, 'k--', linewidth=1, alpha=0.7, where='post', label='Basal Rate')

    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Insulin (U per 15min)')
    ax2.set_title('Calibrated MPC Insulin Delivery')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Insulin deviations
    ax3 = axes[0, 2]
    if results['insulin_deviations']:
        # Ensure matching array lengths
        insulin_dev_times = insulin_times[:len(results['insulin_deviations'])]
        ax3.plot(insulin_dev_times, results['insulin_deviations'], 'purple', linewidth=2, marker='o', markersize=3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Insulin Deviation (U)')
        ax3.set_title('Control Actions (U - Ū)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction accuracy
    ax4 = axes[1, 0]
    if results['predictions']:
        # Show a few predictions
        for i in range(0, min(len(results['predictions']), 16), 4):
            pred_start_time = insulin_times[i]
            pred_times = [pred_start_time + j*0.25 for j in range(8)]
            pred_values = results['predictions'][i]

            ax4.plot(pred_times, pred_values, '--', alpha=0.6, linewidth=1)

        ax4.plot(glucose_times, results['glucose'], 'b-', linewidth=2, label='Actual')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Glucose (mg/dL)')
        ax4.set_title('MPC Predictions vs Actual')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Cost evolution
    ax5 = axes[1, 1]
    if results['costs']:
        # Ensure matching array lengths
        cost_times = insulin_times[:len(results['costs'])]
        ax5.plot(cost_times, results['costs'], 'green', linewidth=2)
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('MPC Cost')
        ax5.set_title('MPC Cost Function')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Performance summary
    ax6 = axes[1, 2]
    summary = results['summary']
    metrics = ['TIR\n70-180', 'TIR\n70-250', 'Mean\nBG', 'CV\n(%)', 'Total\nInsulin', 'Hypo\nEvents']
    values = [
        summary['time_in_range_70_180'],
        summary['time_in_range_70_250'],
        summary['mean_glucose'],
        summary['glucose_cv'],
        summary['total_insulin'],
        summary['hypoglycemia_events']
    ]

    colors = ['green', 'lightgreen', 'blue', 'orange', 'red', 'purple']
    bars = ax6.bar(range(len(metrics)), values, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(metrics)))
    ax6.set_xticklabels(metrics)
    ax6.set_title('Performance Summary')
    ax6.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('mpc_results.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()