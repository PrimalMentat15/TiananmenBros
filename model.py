#!/usr/bin/env python3
"""
Ultra Winning MotoGP Model - 15 Minute Fast Training
Optimized for Ryzen 9 6900HS + RTX 3060 6GB
Uses train_cleaned.csv for training and val.csv for RMSE calculation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Conditional imports for advanced models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost: AVAILABLE")
except ImportError:
    XGB_AVAILABLE = False
    print("❌ XGBoost: NOT AVAILABLE")

try:
    import catboost as cb
    CAT_AVAILABLE = True
    print("✅ CatBoost: AVAILABLE")
except ImportError:
    CAT_AVAILABLE = False
    print("❌ CatBoost: NOT AVAILABLE")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("✅ LightGBM: AVAILABLE")
except ImportError:
    LGB_AVAILABLE = False
    print("❌ LightGBM: NOT AVAILABLE")

class FastUltraWinningModel:
    def __init__(self):
        self.target_col = 'Lap_Time_Seconds'
        self.models = {}
        self.circuit_stats = {}
        self.rider_circuit_stats = {}
        self.weather_circuit_stats = {}
        self.track_circuit_stats = {}
        self.grid_stats = {}
        self.gpu_available = self._check_gpu()
    
    def _check_gpu(self):
        """Check GPU availability"""
        print("🔍 Checking GPU availability...")
        gpu_available = False
        
        if XGB_AVAILABLE:
            try:
                import xgboost as xgb
                test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                X_test = np.random.rand(10, 5)
                y_test = np.random.rand(10)
                test_model.fit(X_test, y_test)
                gpu_available = True
                print("✅ XGBoost GPU: AVAILABLE")
            except Exception as e:
                print(f"❌ XGBoost GPU: NOT AVAILABLE ({str(e)[:50]}...)")
        
        if CAT_AVAILABLE:
            try:
                import catboost as cb
                test_model = cb.CatBoostRegressor(task_type='GPU', iterations=1, verbose=False)
                X_test = np.random.rand(10, 5)
                y_test = np.random.rand(10)
                test_model.fit(X_test, y_test)
                gpu_available = True
                print("✅ CatBoost GPU: AVAILABLE")
            except Exception as e:
                print(f"❌ CatBoost GPU: NOT AVAILABLE ({str(e)[:50]}...)")
        
        return gpu_available
    
    def load_data(self):
        """Load training and validation data"""
        print("📂 Loading data for training and validation...")
        
        self.train_df = pd.read_csv('train_cleaned.csv')
        print(f"✅ Training data: {self.train_df.shape}")
        
        self.val_df = pd.read_csv('val.csv')
        print(f"✅ Validation data: {self.val_df.shape}")
        
        # Verify target column
        if self.target_col in self.train_df.columns:
            print(f"✅ Training target column '{self.target_col}' found")
            print(f"📊 Train target range: {self.train_df[self.target_col].min():.3f} - {self.train_df[self.target_col].max():.3f}")
            print(f"📊 Train target mean: {self.train_df[self.target_col].mean():.3f}")
        
        if self.target_col in self.val_df.columns:
            print(f"✅ Validation target column '{self.target_col}' found")
            print(f"📊 Val target range: {self.val_df[self.target_col].min():.3f} - {self.val_df[self.target_col].max():.3f}")
            print(f"📊 Val target mean: {self.val_df[self.target_col].mean():.3f}")
        else:
            print("❌ Validation data missing target column!")
        
        return self
    
    def fast_feature_engineering(self, df, is_train=True):
        """Fast but effective feature engineering - optimized for 15min training"""
        print("🔧 Fast feature engineering...")
        df = df.copy()
        
        # Essential circuit-based features only
        if is_train:
            self.circuit_stats = df.groupby('circuit_name')[self.target_col].agg([
                'mean', 'std', 'count'
            ]).to_dict()
            
            self.rider_circuit_stats = df.groupby(['rider_name', 'circuit_name'])[self.target_col].agg([
                'mean', 'count'
            ]).to_dict()

        # Apply essential circuit features
        for stat in ['mean', 'std', 'count']:
            df[f'circuit_{stat}'] = df['circuit_name'].map(self.circuit_stats[stat])
            df[f'circuit_{stat}'].fillna(df[f'circuit_{stat}'].median(), inplace=True)
        
        # Essential rider-circuit features
        df['rider_circuit_key'] = df['rider_name'] + '_' + df['circuit_name']
        for stat in ['mean', 'count']:
            df[f'rider_circuit_{stat}'] = df['rider_circuit_key'].map(self.rider_circuit_stats[stat])
            if stat == 'mean':
                df[f'rider_circuit_{stat}'].fillna(df['circuit_mean'], inplace=True)
            else:
                df[f'rider_circuit_{stat}'].fillna(1.0, inplace=True)
        
        # Essential engineered features
        weather_severity = {
            'Clear': 1.0, 'Sunny': 1.0, 'Cloudy': 1.5, 'Overcast': 2.0, 
            'Raining': 4.0, 'Wet': 3.5, 'Drizzle': 2.5
        }
        df['weather_severity'] = df['weather'].map(weather_severity).fillna(2.0)
        
        track_severity = {'Dry': 1.0, 'Damp': 2.5, 'Wet': 4.0}
        df['track_severity'] = df['Track_Condition'].map(track_severity).fillna(1.0)
        
        # Key performance features
        df['temp_humidity_interaction'] = df['Ambient_Temperature_Celsius'] * df['Humidity_%'] / 100
        df['speed_per_km'] = df['Avg_Speed_kmh'] / (df['Circuit_Length_km'] + 0.001)
        df['win_rate'] = df['wins'] / (df['starts'] + 1)
        df['podium_rate'] = df['podiums'] / (df['starts'] + 1)
        df['skill_score'] = (df['win_rate'] * 3 + df['podium_rate'] * 2) / 5
        
        # Grid normalization
        if is_train:
            self.grid_stats = df.groupby('circuit_name')['Grid_Position'].agg(['mean', 'max']).to_dict()
        
        df['circuit_avg_grid'] = df['circuit_name'].map(self.grid_stats['mean']).fillna(15.0)
        df['circuit_max_grid'] = df['circuit_name'].map(self.grid_stats['max']).fillna(30.0)
        df['grid_normalized'] = df['Grid_Position'] / (df['circuit_max_grid'] + 1)
        
        # Add this to your feature engineering:
        categorical_cols = ['circuit_name', 'rider_name', 'weather', 'Track_Condition', 
                           'Tire_Compound_Front', 'Tire_Compound_Rear']

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def prepare_features(self, df):
        """Prepare optimized feature set"""
        print("🎯 Preparing features...")
        
        feature_columns = [
            # Core features
            'Circuit_Length_km', 'Laps', 'Grid_Position', 'Avg_Speed_kmh',
            'Humidity_%', 'Championship_Points', 'Championship_Position',
            'Corners_per_Lap', 'Tire_Degradation_Factor_per_Lap',
            'Pit_Stop_Duration_Seconds', 'Ambient_Temperature_Celsius',
            'Track_Temperature_Celsius', 'starts', 'finishes', 'wins', 'podiums',
            
            # Essential circuit features
            'circuit_mean', 'circuit_std', 'circuit_count',
            'rider_circuit_mean', 'rider_circuit_count',
            
            # Key engineered features
            'weather_severity', 'track_severity', 'temp_humidity_interaction',
            'speed_per_km', 'win_rate', 'podium_rate', 'skill_score',
            'circuit_avg_grid', 'circuit_max_grid', 'grid_normalized'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Handle infinite values properly - replace column by column
        for col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
        
        print(f"📊 Features selected: {len(available_features)}")
        print(f"✅ Missing values: {X.isnull().sum().sum()}")
        
        return X
    
    def train_fast_ensemble(self, X_train, y_train):
        """Train fast but powerful ensemble - optimized for 15 minutes"""
        print("🚀 Training FAST ensemble (15min target)...")
        
        # Fast but effective models - optimized for Ryzen 9 6900HS
        models_config = {
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=None,  # Allow deeper trees
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=16
            ),
            'et': ExtraTreesRegressor(
                n_estimators=500,  # Increased from 300
                max_depth=20,      # Increased from 15
                min_samples_split=2, # Reverted to default for better fit
                min_samples_leaf=1,  # Reverted to default for better fit
                random_state=42,
                n_jobs=16  # Use all 16 threads
            ),
            'ridge': Ridge(alpha=0.1)
        }
        
        # Add GPU-accelerated models if available
        if XGB_AVAILABLE:
            xgb_params = {
                'n_estimators': 500,  # Reduced from 2000
                'max_depth': 6,       # Reduced from 8
                'learning_rate': 0.1, # Increased from 0.05 for faster training
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 16
            }
            if self.gpu_available:
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['gpu_id'] = 0
                print("🚀 XGBoost using RTX 3060 GPU acceleration!")
            models_config['xgb'] = xgb.XGBRegressor(**xgb_params)

        if CAT_AVAILABLE:
            cat_params = {
                'iterations': 500,    # Reduced from 2000
                'depth': 6,           # Reduced from 8
                'learning_rate': 0.1, # Increased from 0.05
                'verbose': False,
                'random_seed': 42,
            }
            if self.gpu_available:
                cat_params['task_type'] = 'GPU'
                print("🚀 CatBoost using RTX 3060 GPU acceleration!")
            else:
                cat_params['thread_count'] = 16  # Use all CPU threads
            models_config['cat'] = cb.CatBoostRegressor(**cat_params)

        if LGB_AVAILABLE:
            models_config['lgb'] = lgb.LGBMRegressor(
                n_estimators=500,     # Reduced from 2000
                max_depth=6,          # Reduced from 8
                learning_rate=0.1,    # Increased from 0.05
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=16,            # Use all CPU threads
                verbose=-1
            )
        
        print(f"🔥 Training {len(models_config)} models with hardware optimization...")
        print(f"💻 CPU: Ryzen 9 6900HS (16 threads)")
        print(f"🎮 GPU: RTX 3060 6GB")
        
        # Train models
        for name, model in models_config.items():
            print(f"\n🎯 Training {name.upper()}...")
            model.fit(X_train, y_train)
            
            # Training RMSE
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            print(f"✅ {name.upper()} Train RMSE: {train_rmse:.6f}")
            
            self.models[name] = model
        
        return self
    
    def predict_ensemble(self, X):
        """Generate ensemble predictions"""
        print("🎯 Generating ensemble predictions...")
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            print(f"✅ {name.upper()} predictions: {pred.min():.3f} - {pred.max():.3f}")
        
        # Ensemble prediction
        ensemble_pred = np.mean(predictions, axis=0)
        
        print(f"🏆 ENSEMBLE predictions: {ensemble_pred.min():.3f} - {ensemble_pred.max():.3f}")
        print(f"🏆 ENSEMBLE mean: {ensemble_pred.mean():.3f}")
        print(f"🏆 ENSEMBLE unique values: {len(np.unique(ensemble_pred))}")
        
        return ensemble_pred
    
    def run_fast_validation_pipeline(self):
        """Execute fast validation pipeline"""
        print("🏁 Starting FAST VALIDATION PIPELINE (15min target)...")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Fast feature engineering on training data
        print("\n" + "="*50)
        print("🔧 FAST TRAINING DATA FEATURE ENGINEERING")
        print("="*50)
        train_engineered = self.fast_feature_engineering(self.train_df, is_train=True)
        X_train = self.prepare_features(train_engineered)
        y_train = self.train_df[self.target_col]
        
        print(f"✅ Training features shape: {X_train.shape}")
        print(f"✅ Training target shape: {y_train.shape}")
        
        # Train fast ensemble
        print("\n" + "="*50)
        print("🚀 FAST ENSEMBLE TRAINING")
        print("="*50)
        self.train_fast_ensemble(X_train, y_train)
        
        # Fast feature engineering on validation data
        print("\n" + "="*50)
        print("🔧 FAST VALIDATION DATA FEATURE ENGINEERING")
        print("="*50)
        val_engineered = self.fast_feature_engineering(self.val_df, is_train=False)
        X_val = self.prepare_features(val_engineered)
        
        print(f"✅ Validation features shape: {X_val.shape}")
        
        # Generate predictions on validation data
        print("\n" + "="*50)
        print("🎯 VALIDATION PREDICTION")
        print("="*50)
        val_predictions = self.predict_ensemble(X_val)
        
        # Calculate RMSE on validation data
        print("\n" + "="*50)
        print("📊 VALIDATION RMSE CALCULATION")
        print("="*50)
        
        if self.target_col in self.val_df.columns:
            y_val_true = self.val_df[self.target_col]
            val_rmse = np.sqrt(mean_squared_error(y_val_true, val_predictions))
            
            print(f"🏆 VALIDATION RMSE: {val_rmse:.6f}")
            print(f"📊 Validation predictions range: {val_predictions.min():.3f} - {val_predictions.max():.3f}")
            print(f"📊 Validation actual range: {y_val_true.min():.3f} - {y_val_true.max():.3f}")
            print(f"📊 Unique predictions: {len(np.unique(val_predictions))}")
            
            # Additional metrics
            mae = np.mean(np.abs(y_val_true - val_predictions))
            print(f"📊 Mean Absolute Error: {mae:.6f}")
            
            # Check prediction diversity
            identical_count = np.sum(val_predictions == val_predictions[0])
            identical_percentage = (identical_count / len(val_predictions)) * 100
            print(f"📊 Identical predictions: {identical_count}/{len(val_predictions)} ({identical_percentage:.1f}%)")
            
            print("\n🏆 FAST VALIDATION RESULTS:")
            print("="*80)
            print(f"   📊 RMSE: {val_rmse:.6f}")
            print(f"   📊 MAE: {mae:.6f}")
            print(f"   📊 Unique predictions: {len(np.unique(val_predictions))}")
            print(f"   📊 Identical predictions: {identical_percentage:.1f}%")
            print(f"   💻 Hardware: Ryzen 9 6900HS + RTX 3060")
            print(f"   ⏱️ Target: <15 minutes")
            print("   🏆 SUCCESS!")
            print("="*80)
            
            return val_predictions, val_rmse
        else:
            print("❌ Validation data does not contain target column!")
            return val_predictions, None

if __name__ == "__main__":
    model = FastUltraWinningModel()
    predictions, rmse = model.run_fast_validation_pipeline()
    print("🏁 Fast validation pipeline completed successfully!") 