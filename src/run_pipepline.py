#Run pipeine 

import sys
import os
from pathlib import Path

def run_pipeline(use_sample_data=True):
    """
    Run complete ML pipeline for stress detection
    
    Args:
        use_sample_data: If True, use synthetic sample data. 
                        If False, attempt to use real WESAD data.
    """
    print("="*70)
    print("STRESS DETECTION ML PIPELINE")
    print("="*70)
    print("\nThis pipeline implements:")
    print("  1.Data loading and preprocessing")
    print("  2.Feature extraction from physiological signals")
    print("  3.Multiple ML model training")
    print("  4.Leave-One-Subject-Out (LOSO) cross-validation")
    print("  5.Model evaluation and comparison")
    print("  6.Interactive demo")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    from data_loader import WESADDataLoader
    import numpy as np
    
    loader = WESADDataLoader()
    
    if use_sample_data:
        print("\nUsing synthetic sample data for demo purposes...")
        X, y, subject_ids = loader.create_sample_dataset()
    else:
        print("\nAttempting to load real WESAD data...")
        try:
            X, y, subject_ids = loader.prepare_dataset()
            print("✓ Real WESAD data loaded successfully!")
        except FileNotFoundError:
            print("\n⚠ Real data not found. Using sample data instead.")
            print("To use real data, download WESAD from:")
            print("https://archive.ics.uci.edu/ml/datasets/WESAD+")
            X, y, subject_ids = loader.create_sample_dataset()
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    np.save(output_dir / 'subject_ids.npy', subject_ids)
    
    print(f"\n✓ Data preprocessing complete!")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Subjects: {len(np.unique(subject_ids))}")
    print(f"  Stress samples: {np.sum(y == 1)}")
    print(f"  Non-stress samples: {np.sum(y == 0)}")
    
    # Step 2: Train and evaluate models
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    from train_models import StressDetectionModels
    
    trainer = StressDetectionModels()
    
    # Evaluate all models with LOSO CV
    all_results = trainer.evaluate_all_models(X, y, subject_ids)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    trainer.plot_results_comparison(all_results)
    
    # Generate report
    print("Generating evaluation report...")
    trainer.generate_report(all_results)
    
    # Train final model
    best_model_name = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nTraining final model: {best_model_name}")
    final_model, scaler = trainer.train_final_model(X, y, best_model_name)
    
    # Save model
    import joblib
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(final_model, model_dir / 'stress_detector.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    print(f"✓ Final model saved to {model_dir}")
    
    # Step 3: Run demo
    print("\n" + "="*70)
    print("STEP 3: INTERACTIVE DEMO")
    print("="*70)
    
    from demo import StressDetectorDemo
    
    demo = StressDetectorDemo()
    results = demo.run_interactive_demo()
    demo.visualize_predictions(results)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  data/processed/       - Preprocessed dataset")
    print("  models/              - Trained models")
    print("  results/             - Evaluation results and plots")
    print("\nKey outputs:")
    print("  results/model_comparison.png     - Model performance comparison")
    print("  results/demo_predictions.png     - Demo predictions visualization")
    print("  results/evaluation_report.txt    - Detailed evaluation report")
    print("\nBest performing model:")
    best_acc = all_results[best_model_name]['accuracy']
    print(f"  ** {best_model_name} **")
    print(f"   Accuracy: {best_acc:.1%}")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    # Check if user wants to use real data
    use_sample = True
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real-data":
        use_sample = False
    
    if use_sample:
        print("\nRunning with SAMPLE DATA for demo")
        print("   To use real WESAD data, run: python run_pipeline.py --real-data")
        print()
    
    # Run pipeline
    results = run_pipeline(use_sample_data=use_sample)
    
    print("\n All done! Check the results/ folder for outputs.")