"""
Demo script for stress detection 
"""

import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class StressDetectorDemo:
    """
    Demo interface for stress detection model
    """
    
    def __init__(self, model_path='models/stress_detector.pkl', 
                 scaler_path='models/scaler.pkl'):
        """Load trained model and scaler"""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
        
        # Construct full paths
        full_model_path = os.path.join(project_root, model_path)
        full_scaler_path = os.path.join(project_root, scaler_path)
        
        # Load model and scaler
        self.model = joblib.load(full_model_path)
        self.scaler = joblib.load(full_scaler_path)
        
        # Feature names
        self.feature_names = [
            'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
            'bvp_mean', 'bvp_std', 'bvp_min', 'bvp_max',
            'temp_mean', 'temp_std'
        ]
        
        print("Stress Detector Model Loaded!")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Number of features: {len(self.feature_names)}")
        
    def predict_stress(self, features):
        """
        Predict stress level from physiological features
        
        Args:
            features: Dictionary or array of feature values
            
        Returns:
            prediction: 0 (non-stress) or 1 (stress)
            probability: Confidence of prediction
        """
        # Convert to array if dict
        if isinstance(features, dict):
            feature_array = np.array([features[name] for name in self.feature_names])
        else:
            feature_array = np.array(features)
        
        # Reshape for single prediction
        feature_array = feature_array.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            probability = probabilities[prediction]
        else:
            probability = 1.0  # SVM doesn't have predict_proba by default
        
        return prediction, probability
    
    def generate_sample_features(self, stress_level='normal'):
        """
        Generate sample physiological features
        
        Args:
            stress_level: 'normal', 'mild_stress', 'high_stress'
            
        Returns:
            Dictionary of features
        """
        np.random.seed(42)
        
        if stress_level == 'normal':
            # Normal baseline values
            features = {
                'eda_mean': np.random.uniform(0.2, 0.5),
                'eda_std': np.random.uniform(0.05, 0.15),
                'eda_min': np.random.uniform(0.1, 0.3),
                'eda_max': np.random.uniform(0.3, 0.6),
                'eda_range': np.random.uniform(0.1, 0.3),
                'bvp_mean': np.random.uniform(-50, 50),
                'bvp_std': np.random.uniform(20, 40),
                'bvp_min': np.random.uniform(-100, -50),
                'bvp_max': np.random.uniform(50, 100),
                'temp_mean': np.random.uniform(32, 33),
                'temp_std': np.random.uniform(0.1, 0.3)
            }
        elif stress_level == 'mild_stress':
            # Mild stress - slightly elevated values
            features = {
                'eda_mean': np.random.uniform(0.5, 0.8),
                'eda_std': np.random.uniform(0.15, 0.25),
                'eda_min': np.random.uniform(0.3, 0.5),
                'eda_max': np.random.uniform(0.7, 1.0),
                'eda_range': np.random.uniform(0.3, 0.5),
                'bvp_mean': np.random.uniform(30, 80),
                'bvp_std': np.random.uniform(35, 50),
                'bvp_min': np.random.uniform(-80, -40),
                'bvp_max': np.random.uniform(80, 120),
                'temp_mean': np.random.uniform(33, 34),
                'temp_std': np.random.uniform(0.2, 0.4)
            }
        else:  # high_stress
            # High stress - significantly elevated values
            features = {
                'eda_mean': np.random.uniform(0.8, 1.2),
                'eda_std': np.random.uniform(0.25, 0.4),
                'eda_min': np.random.uniform(0.5, 0.7),
                'eda_max': np.random.uniform(1.0, 1.5),
                'eda_range': np.random.uniform(0.5, 0.8),
                'bvp_mean': np.random.uniform(60, 100),
                'bvp_std': np.random.uniform(45, 60),
                'bvp_min': np.random.uniform(-60, -20),
                'bvp_max': np.random.uniform(100, 150),
                'temp_mean': np.random.uniform(34, 35),
                'temp_std': np.random.uniform(0.3, 0.5)
            }
        
        return features
    
    def run_interactive_demo(self):
        """
        Run interactive demo with sample scenarios
        """
        print("\n" + "="*60)
        print("STRESS DETECTION DEMO")
        print("="*60)
        
        scenarios = [
            ('Normal Baseline', 'normal'),
            ('Mild Stress (e.g., focused work)', 'mild_stress'),
            ('High Stress (e.g., public speaking)', 'high_stress')
        ]
        
        results = []
        
        for scenario_name, stress_level in scenarios:
            print(f"\n{scenario_name}:")
            print("-" * 60)
            
            # Generate features
            features = self.generate_sample_features(stress_level)
            
            # Display some key features
            print("Physiological Readings:")
            print(f"  Electrodermal Activity (EDA): {features['eda_mean']:.3f}")
            print(f"  Blood Volume Pulse (BVP): {features['bvp_mean']:.2f}")
            print(f"  Skin Temperature: {features['temp_mean']:.2f}°C")
            
            # Predict
            prediction, probability = self.predict_stress(features)
            
            # Display result
            status = "STRESS DETECTED" if prediction == 1 else "NO STRESS"
            print(f"\n  Prediction: {status}")
            print(f"  Confidence: {probability:.1%}")
            
            results.append({
                'scenario': scenario_name,
                'prediction': prediction,
                'probability': probability,
                'features': features
            })
        
        print("\n" + "="*60)
        
        return results
    
    def visualize_predictions(self, results, save_path='results/demo_predictions.png'):
        """
        Visualize demo predictions
        
        Args:
            results: Results from run_interactive_demo
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Predictions
        ax = axes[0]
        scenarios = [r['scenario'] for r in results]
        predictions = [r['prediction'] for r in results]
        probabilities = [r['probability'] for r in results]
        
        colors = ['green' if p == 0 else 'red' for p in predictions]
        bars = ax.barh(scenarios, probabilities, color=colors, alpha=0.7)
        
        ax.set_xlabel('Confidence')
        ax.set_title('Stress Detection Results')
        ax.set_xlim([0, 1])
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        #Add labels
        for i, (bar, pred) in enumerate(zip(bars, predictions)):
            label = 'STRESS' if pred == 1 else 'NO STRESS'
            ax.text(0.5, i, f'  {label}', va='center', fontweight='bold', color='white')
        
        #Plot 2: Feature comparison
        ax = axes[1]
        
        #Compare EDA mean across scenarios
        eda_means = [r['features']['eda_mean'] for r in results]
        bvp_means = [r['features']['bvp_mean'] for r in results]
        temp_means = [r['features']['temp_mean'] for r in results]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        # Normalize for comparison
        eda_norm = [(e - min(eda_means))/(max(eda_means) - min(eda_means)) for e in eda_means]
        bvp_norm = [(b - min(bvp_means))/(max(bvp_means) - min(bvp_means)) for b in bvp_means]
        temp_norm = [(t - min(temp_means))/(max(temp_means) - min(temp_means)) for t in temp_means]
        
        ax.bar(x - width, eda_norm, width, label='EDA (normalized)', alpha=0.8)
        ax.bar(x, bvp_norm, width, label='BVP (normalized)', alpha=0.8)
        ax.bar(x + width, temp_norm, width, label='Temp (normalized)', alpha=0.8)
        
        ax.set_ylabel('Normalized Value')
        ax.set_title('Physiological Features by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        
        plt.show()
    
    def predict_custom(self, eda_mean, bvp_mean, temp_mean):
        
        #Make prediction with custom values (simplified interface)
        
        #Args:
            #eda_mean: Electrodermal activity mean
            #bvp_mean: Blood volume pulse mean  
            #temp_mean: Temperature mean
            
        #Returns:
            #Prediction and probability
        
        # reate feature dict with reasonable defaults
        features = {
            'eda_mean': eda_mean,
            'eda_std': eda_mean * 0.2,  #Assume std is 20% of mean
            'eda_min': eda_mean * 0.7,
            'eda_max': eda_mean * 1.3,
            'eda_range': eda_mean * 0.6,
            'bvp_mean': bvp_mean,
            'bvp_std': abs(bvp_mean) * 0.5,
            'bvp_min': bvp_mean - 50,
            'bvp_max': bvp_mean + 50,
            'temp_mean': temp_mean,
            'temp_std': 0.2
        }
        
        return self.predict_stress(features)


def main():
    """Run the demo"""
    #Initialize demo
    demo = StressDetectorDemo()
    
    #Run interactive demo
    results = demo.run_interactive_demo()
    
    #Visualize results
    demo.visualize_predictions(results)
    
    #Example custom prediction
    print("\n" + "="*60)
    print("CUSTOM PREDICTION EXAMPLE")
    print("="*60)
    print("\nSimulating: Person giving a presentation (high stress)")
    prediction, probability = demo.predict_custom(
        eda_mean=1.0,  # High EDA
        bvp_mean=85,   # Elevated heart rate
        temp_mean=34.5  # Elevated temperature
    )
    
    status = "STRESS DETECTED" if prediction == 1 else "NO STRESS"
    print(f"Prediction: {status}")
    print(f"Confidence: {probability:.1%}")
    
    print("\n" + "="*60)
    print("Demo complete!")


if __name__ == "__main__":
    main()