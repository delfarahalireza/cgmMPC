#!/usr/bin/env python3
"""
Temporal-Aware data generator for LSTM-MPC training using SimGlucose simulator
Implements proper forward chaining validation with circadian pattern preservation
Based on Aiello et al. 2023 methodology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import os
import sys
from pathlib import Path

# Add SimGlucose to path
sys.path.append('./simglucose')

# SimGlucose imports
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action
import warnings
warnings.filterwarnings('ignore')

class TemporalDataGenerator:
    """
    Generate temporally-aware training data for FT and GT predictors
    Key principles:
    1. Preserve circadian patterns and meal timing
    2. Forward chaining validation (no future data leakage)
    3. Controlled scenario generation
    4. Complete state information for GT training
    """
    
    def __init__(self, sampling_minutes: int = 15):
        self.sampling_minutes = sampling_minutes
        self.sampling_hours = sampling_minutes / 60.0
        
        # Standard meal schedule (realistic circadian timing)
        self.meal_schedule = {
            'breakfast': {'time': 8.0, 'carbs': 45},   # 8:00 AM, 45g
            'lunch': {'time': 13.0, 'carbs': 60},      # 1:00 PM, 60g  
            'dinner': {'time': 19.0, 'carbs': 75}      # 7:00 PM, 75g
        }
        
        # Realistic basal rates with circadian variation
        self.basal_schedule = {
            0: 0.8,   # Midnight: 0.8 U/hr
            3: 0.7,   # 3 AM: 0.7 U/hr (lowest)
            6: 1.2,   # 6 AM: 1.2 U/hr (dawn phenomenon)
            9: 1.0,   # 9 AM: 1.0 U/hr
            12: 0.9,  # Noon: 0.9 U/hr
            15: 0.8,  # 3 PM: 0.8 U/hr
            18: 0.9,  # 6 PM: 0.9 U/hr
            21: 0.8   # 9 PM: 0.8 U/hr
        }
    
    def generate_temporal_training_data(self, 
                                      n_patients: int = 10,
                                      days_per_patient: int = 7,
                                      train_val_split: float = 0.7) -> Dict:
        """
        Generate temporal training data with forward chaining validation
        
        Args:
            n_patients: Number of virtual patients
            days_per_patient: Days of simulation per patient
            train_val_split: Fraction for training (rest for validation)
        
        Returns:
            Dictionary with temporally-organized training data
        """
        
        print("=" * 60)
        print("TEMPORAL SIMGLUCOSE DATA GENERATION")
        print("=" * 60)
        print(f"Patients: {n_patients}")
        print(f"Days per patient: {days_per_patient}")
        print(f"Sampling: Every {self.sampling_minutes} minutes")
        print(f"Train/Val split: {train_val_split:.1%}/{1-train_val_split:.1%}")
        
        # Generate data for both scenarios
        scenario_1_data = self._generate_scenario_data(
            n_patients, days_per_patient, scenario_type='basal_only'
        )
        
        scenario_2_data = self._generate_scenario_data(
            n_patients, days_per_patient, scenario_type='basal_bolus'
        )
        
        # Apply forward chaining split
        train_data, val_data = self._apply_forward_chaining_split(
            scenario_1_data, scenario_2_data, train_val_split
        )
        
        # Create sequences with temporal integrity
        training_sequences = self._create_temporal_sequences(train_data)
        validation_sequences = self._create_temporal_sequences(val_data)
        
        # Organize final dataset
        temporal_data = {
            'training': training_sequences,
            'validation': validation_sequences,
            'metadata': {
                'n_patients': n_patients,
                'days_per_patient': days_per_patient,
                'sampling_minutes': self.sampling_minutes,
                'train_val_split': train_val_split,
                'meal_schedule': self.meal_schedule,
                'basal_schedule': self.basal_schedule,
                'generation_time': datetime.now().isoformat()
            }
        }
        
        return temporal_data
    
    def _generate_scenario_data(self, n_patients: int, days: int, 
                              scenario_type: str) -> Dict:
        """Generate controlled scenario data"""
        
        print(f"\nGenerating {scenario_type} scenario data...")
        
        scenario_data = {}
        
        # Use selected adult patients: 1, 2, 4, 5, 8, 9, 10
        selected_adults = [1, 2, 4, 5, 8, 9, 10]
        
        for i in range(n_patients):
            print(f"  Patient {i + 1}/{n_patients}...", end="")
            
            # Use adult patients from the selected list
            if i < len(selected_adults):
                adult_id = selected_adults[i]
                patient_name = f'adult#{adult_id:03d}'
            else:
                # If we need more patients than selected, cycle through the list
                adult_id = selected_adults[i % len(selected_adults)]
                patient_name = f'adult#{adult_id:03d}'
            
            # Create virtual patient
            patient = T1DPatient.withName(patient_name)
            sensor = CGMSensor.withName('Dexcom', seed=i)
            pump = InsulinPump.withName('Insulet')
            
            # Create custom scenario with realistic meals
            scenario = self._create_meal_scenario(days)
            
            # Create simulation environment
            env = T1DSimEnv(patient, sensor, pump, scenario)
            
            # Create controller based on scenario
            if scenario_type == 'basal_only':
                controller = BasalOnlyController(self.basal_schedule)
            else:  # basal_bolus
                controller = BasalBolusController(self.basal_schedule, self.meal_schedule)
            
            # Simulate
            sim_obj = SimObj(env, controller, timedelta(days=days), 
                           animate=False, path=f'./temp_results_{scenario_type}_{i}')
            results = sim(sim_obj)
            
            # Process simulation results
            patient_data = self._process_simulation_results(results, i)
            scenario_data[f'patient_{i}'] = patient_data
            
            print(" ✓")
        
        return scenario_data
    
    def _create_meal_scenario(self, days: int) -> CustomScenario:
        """Create realistic meal scenario with circadian timing"""
        
        scenario_data = []
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for day in range(days):
            for meal_name, meal_info in self.meal_schedule.items():
                meal_time_hours = day * 24 + meal_info['time']
                # Convert to (time_from_start_in_hours, carbs) format
                scenario_data.append((meal_time_hours, meal_info['carbs']))
        
        return CustomScenario(start_time=start_time, scenario=scenario_data)
    
    def _process_simulation_results(self, results, patient_id: int) -> pd.DataFrame:
        """Process simulation results into structured dataframe"""
        
        # Results is already a DataFrame from SimGlucose
        df = results.copy().reset_index()
        
        # Rename columns for consistency
        df = df.rename(columns={
            'Time': 'timestamp',
            'BG': 'glucose',
            'CGM': 'cgm', 
            'CHO': 'carbs'
        })
        
        # Add derived columns
        df['patient_id'] = patient_id
        df['time_of_day'] = (df['timestamp'].dt.hour + df['timestamp'].dt.minute/60.0) % 24
        
        # Add insulin columns from SimGlucose results
        # SimGlucose records total insulin delivery in 'insulin' column (U/min)
        if 'insulin' in df.columns:
            df['insulin_total'] = df['insulin']
            df['insulin_basal'] = df['insulin']  # For now, treat as total insulin
            df['insulin_bolus'] = 0.0  # Will be calculated as difference in scenarios
        else:
            df['insulin_total'] = 0.0
            df['insulin_basal'] = 0.0
            df['insulin_bolus'] = 0.0
        
        # Resample to uniform 15-minute grid if needed
        if self.sampling_minutes != 3:  # SimGlucose default is 3 minutes
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.resample(f'{self.sampling_minutes}T').mean()
            df = df.fillna(method='ffill')
            df = df.reset_index()
            df['patient_id'] = patient_id
        
        return df
    
    def _apply_forward_chaining_split(self, scenario_1: Dict, scenario_2: Dict,
                                    split_ratio: float) -> Tuple[Dict, Dict]:
        """Apply forward chaining validation split preserving temporal order"""
        
        print(f"\nApplying forward chaining split ({split_ratio:.1%} train)...")
        
        train_data = {'scenario_1': {}, 'scenario_2': {}}
        val_data = {'scenario_1': {}, 'scenario_2': {}}
        
        # Process each patient separately to maintain temporal integrity
        for patient_key in scenario_1.keys():
            
            # Get patient data from both scenarios
            s1_data = scenario_1[patient_key]
            s2_data = scenario_2[patient_key]
            
            # Find split point (temporal, not random)
            n_samples = len(s1_data)
            split_idx = int(n_samples * split_ratio)
            
            # Split preserving temporal order
            train_data['scenario_1'][patient_key] = s1_data.iloc[:split_idx].copy()
            train_data['scenario_2'][patient_key] = s2_data.iloc[:split_idx].copy()
            
            val_data['scenario_1'][patient_key] = s1_data.iloc[split_idx:].copy()
            val_data['scenario_2'][patient_key] = s2_data.iloc[split_idx:].copy()
        
        # Print split statistics
        total_train_samples = sum(len(data) for data in train_data['scenario_1'].values())
        total_val_samples = sum(len(data) for data in val_data['scenario_1'].values())
        
        print(f"  Training samples: {total_train_samples:,}")
        print(f"  Validation samples: {total_val_samples:,}")
        print(f"  Actual split: {total_train_samples/(total_train_samples+total_val_samples):.1%}")
        
        return train_data, val_data
    
    def _create_temporal_sequences(self, data: Dict) -> Dict:
        """Create training sequences preserving temporal order"""
        
        print("Creating temporal sequences...")
        
        # Sequence parameters
        lookback_hours = 6  # 6 hours of history
        prediction_horizon = 8  # 8 steps ahead (2 hours)
        lookback_steps = int(lookback_hours / self.sampling_hours)
        
        sequences = {'scenario_1': {}, 'scenario_2': {}}
        
        for scenario_name in ['scenario_1', 'scenario_2']:
            X_sequences = []
            y_sequences = []
            patient_ids = []
            timestamps = []
            
            for patient_key, patient_data in data[scenario_name].items():
                if len(patient_data) < lookback_steps + prediction_horizon:
                    continue  # Skip patients with insufficient data
                
                patient_id = patient_data['patient_id'].iloc[0]
                
                # Create overlapping windows in temporal order
                for i in range(lookback_steps, len(patient_data) - prediction_horizon):
                    
                    # Create feature vector (state at time i)
                    features = self._create_state_vector(patient_data, i, lookback_steps)
                    
                    # Create target sequence (glucose predictions)
                    targets = []
                    for j in range(1, prediction_horizon + 1):
                        targets.append(patient_data.iloc[i + j]['cgm'])
                    
                    X_sequences.append(features)
                    y_sequences.append(targets)
                    patient_ids.append(patient_id)
                    timestamps.append(patient_data.iloc[i]['timestamp'])
            
            sequences[scenario_name] = {
                'X': np.array(X_sequences),
                'y': np.array(y_sequences),
                'patient_ids': np.array(patient_ids),
                'timestamps': timestamps
            }
            
            print(f"  {scenario_name}: {len(X_sequences):,} sequences")
        
        return sequences
    
    def _create_state_vector(self, data: pd.DataFrame, current_idx: int, 
                           lookback_steps: int) -> np.ndarray:
        """Create state vector"""
        
        features = []
        
        # CGM history: current and past lookback_steps
        for j in range(lookback_steps + 1):
            idx = current_idx - j
            if idx >= 0:
                features.append(data.iloc[idx]['cgm'])
            else:
                features.append(100.0)  # Default glucose value
        
        # Insulin history: past lookback_steps (not including current)
        for j in range(1, lookback_steps + 1):
            idx = current_idx - j
            if idx >= 0:
                # Use total insulin delivery recorded by SimGlucose
                total_insulin = data.iloc[idx]['insulin_total']
                features.append(total_insulin)
            else:
                features.append(0.0)
        
        # Carbohydrate history: current and past lookback_steps
        for j in range(lookback_steps + 1):
            idx = current_idx - j
            if idx >= 0:
                features.append(data.iloc[idx]['carbs'])
            else:
                features.append(0.0)
        
        # Time-of-day feature (circadian information)
        features.append(data.iloc[current_idx]['time_of_day'])
        
        return np.array(features)


class BasalOnlyController(Controller):
    """Scenario-I: Basal-only controller with circadian variation"""
    
    def __init__(self, basal_schedule: Dict):
        self.basal_schedule = basal_schedule
        
    def policy(self, observation, reward, done, **info):
        """Return basal insulin based on time of day"""
        
        current_time = info.get('time', datetime.now())
        hour = current_time.hour
        
        # Find appropriate basal rate (U/hr)
        basal_rate = 0.8  # Default
        for schedule_hour in sorted(self.basal_schedule.keys(), reverse=True):
            if hour >= schedule_hour:
                basal_rate = self.basal_schedule[schedule_hour]
                break
        
        # Convert to U/min and return Action
        return Action(basal=basal_rate / 60.0, bolus=0.0)
    
    def reset(self):
        pass


class BasalBolusController(Controller):
    """Scenario-II: Basal + meal bolus controller with improved parameters"""
    
    def __init__(self, basal_schedule: Dict, meal_schedule: Dict):
        self.basal_schedule = basal_schedule
        self.meal_schedule = meal_schedule
        self.last_bolus_time = {}
        # Improved parameters from testing
        self.icr = 6.0  # g/U (more aggressive than 15.0)
        self.bolus_window = 0.5  # hours (wider than 0.25)
        
    def policy(self, observation, reward, done, **info):
        """Return basal + meal bolus insulin"""
        
        current_time = info.get('time', datetime.now())
        hour = current_time.hour + current_time.minute / 60.0
        
        # Get basal rate (U/hr)
        basal_rate = 0.8
        for schedule_hour in sorted(self.basal_schedule.keys(), reverse=True):
            if current_time.hour >= schedule_hour:
                basal_rate = self.basal_schedule[schedule_hour]
                break
        
        # Check for meal bolus with improved parameters
        bolus = 0.0
        for meal_name, meal_info in self.meal_schedule.items():
            meal_time = meal_info['time']
            carbs = meal_info['carbs']
            
            # Bolus if within the improved window
            if abs(hour - meal_time) <= self.bolus_window:
                if meal_name not in self.last_bolus_time:
                    # Calculate bolus with improved ICR
                    bolus = carbs / self.icr
                    self.last_bolus_time[meal_name] = current_time
                    break
        
        # Convert to U/min and return Action
        return Action(basal=basal_rate / 60.0, bolus=bolus / 60.0)
    
    def reset(self):
        self.last_bolus_time = {}


def main():
    """Generate temporal training data with improved insulin parameters and adult patients"""
    
    print("IMPROVED TEMPORAL DATA GENERATION")
    print("=" * 60)
    print("Improvements:")
    print("• ICR: 8.0 g/U (more aggressive insulin)")
    print("• Bolus window: 0.5 hours (better meal coverage)")
    print("• Patients: adult#001, #002, #004, #005, #008, #009, #010")
    print("• Expected better glucose ranges and insulin effects")
    print("=" * 60)
    
    generator = TemporalDataGenerator(sampling_minutes=15)
    
    # Generate full dataset with improved parameters
    temporal_data = generator.generate_temporal_training_data(
        n_patients=10,
        days_per_patient=7,
        train_val_split=0.7
    )
    
    # Save data
    output_path = 'cache/temporal_simglucose_data_v2.pkl'
    os.makedirs('cache', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(temporal_data, f)
    
    print(f"\nTemporal data saved to {output_path}")
    
    # Print summary
    train = temporal_data['training']
    val = temporal_data['validation']
    
    print("\n" + "="*60)
    print("TEMPORAL DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Training Scenario-I: {train['scenario_1']['X'].shape}")
    print(f"Training Scenario-II: {train['scenario_2']['X'].shape}")
    print(f"Validation Scenario-I: {val['scenario_1']['X'].shape}")
    print(f"Validation Scenario-II: {val['scenario_2']['X'].shape}")
    print(f"Features per sample: {train['scenario_1']['X'].shape[1]}")
    print(f"Prediction horizon: {train['scenario_1']['y'].shape[1]} steps")
    print("="*60)


if __name__ == "__main__":
    main()