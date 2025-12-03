

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time


class StressDetectionModels:
    
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        self.results = {}
        self.scaler = StandardScaler()
        
    def evaluate_loso(self, X, y, subject_ids, model_name='Random Forest'):
        """
        Evaluate model using Leave-One-Subject-Out cross-validation
        This ensures the model generalizes to new subjects
        
        Arguments:
            X: Feature matrix
            y: Labels
            subject_ids: Subject ID for each sample
            model_name: Name of model to use
            
        Returns:
            dict: Evaluation metrics
        """
        model = self.models[model_name]
        logo = LeaveOneGroupOut()
        
        all_y_true = []
        all_y_pred = []
        fold_accuracies = []
        
        print(f"\nEvaluating {model_name} with LOSO CV...")
        print(f"Number of subjects: {len(np.unique(subject_ids))}")
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_ids)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            #Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            #Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            #Calculate accuracy for this fold
            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            test_subject = subject_ids[test_idx][0]
            print(f"  Fold {fold+1} (Subject {test_subject}): Accuracy = {fold_acc:.3f}")
        
        #Calculate overall metrics
        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, average='binary', zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, average='binary', zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(all_y_true, all_y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'fold_accuracies': fold_accuracies
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Overall Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Mean Fold Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
        
        return results
    
    def evaluate_all_models(self, X, y, subject_ids):
       #Returns all results for models 
        all_results = {}
        
        print("=" * 60)
        print("EVALUATING ALL MODELS")
        print("=" * 60)
        
        for model_name in self.models.keys():
            start_time = time.time()
            results = self.evaluate_loso(X, y, subject_ids, model_name)
            results['training_time'] = time.time() - start_time
            all_results[model_name] = results
            print(f"Training time: {results['training_time']:.2f} seconds")
            print("-" * 60)
        
        return all_results
    
    def train_final_model(self, X, y, model_name='Random Forest'):
        #Returns trained model and scalar
        print(f"\nTraining final {model_name} model on all data...")
        
        #Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        #Train model
        model = self.models[model_name]
        model.fit(X_scaled, y)
        
        #Calculate training accuracy
        y_pred = model.predict(X_scaled)
        train_acc = accuracy_score(y, y_pred)
        
        print(f"Training accuracy: {train_acc:.3f}")
        
        return model, self.scaler
    
    def plot_results_comparison(self, all_results, save_path='results/model_comparison.png'):
        
        #Figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        model_names = list(all_results.keys())
        accuracies = [all_results[m]['accuracy'] for m in model_names]
        precisions = [all_results[m]['precision'] for m in model_names]
        recalls = [all_results[m]['recall'] for m in model_names]
        f1_scores = [all_results[m]['f1_score'] for m in model_names]
        
        #Plot 1: Metrics comparison
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.2
        
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        #Plot 2: Confusion matrix for best model
        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        cm = best_model[1]['confusion_matrix']
        
        ax = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model[0]}\n(Best Model)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Non-Stress', 'Stress'])
        ax.set_yticklabels(['Non-Stress', 'Stress'])
        
        #Plot 3: Training time comparison
        ax = axes[1, 0]
        training_times = [all_results[m]['training_time'] for m in model_names]
        bars = ax.barh(model_names, training_times, alpha=0.7)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Model Training Time')
        ax.grid(axis='x', alpha=0.3)
        
        #Color bars by performance
        colors = plt.cm.RdYlGn([acc for acc in accuracies])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        #Plot 4: Fold accuracies for best model
        ax = axes[1, 1]
        fold_accs = best_model[1]['fold_accuracies']
        ax.plot(fold_accs, marker='o', linewidth=2, markersize=8)
        ax.axhline(y=np.mean(fold_accs), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(fold_accs):.3f}')
        ax.fill_between(range(len(fold_accs)), 
                        np.mean(fold_accs) - np.std(fold_accs),
                        np.mean(fold_accs) + np.std(fold_accs),
                        alpha=0.2, color='r')
        ax.set_xlabel('Fold (Subject)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'LOSO CV Fold Accuracies - {best_model[0]}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        #Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, all_results, save_path='results/evaluation_report.txt'):
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("STRESS DETECTION MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Dataset: WESAD (Wearable Stress and Affect Detection)\n")
            f.write("Task: Binary Classification (Stress vs Non-Stress)\n")
            f.write("Evaluation: Leave-One-Subject-Out (LOSO) Cross-Validation\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n\n")
            
            #Summary table
            summary_data = []
            for model_name, results in all_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision': f"{results['precision']:.4f}",
                    'Recall': f"{results['recall']:.4f}",
                    'F1-Score': f"{results['f1_score']:.4f}",
                    'Time (s)': f"{results['training_time']:.2f}"
                })
            
            df = pd.DataFrame(summary_data)
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Best model
            best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
            f.write("-" * 70 + "\n")
            f.write(f"BEST MODEL: {best_model[0]}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {best_model[1]['accuracy']:.4f}\n")
            f.write(f"Precision: {best_model[1]['precision']:.4f}\n")
            f.write(f"Recall: {best_model[1]['recall']:.4f}\n")
            f.write(f"F1-Score: {best_model[1]['f1_score']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            cm = best_model[1]['confusion_matrix']
            f.write(f"                Predicted Non-Stress  Predicted Stress\n")
            f.write(f"Actual Non-Stress        {cm[0,0]:5d}              {cm[0,1]:5d}\n")
            f.write(f"Actual Stress            {cm[1,0]:5d}              {cm[1,1]:5d}\n\n")
            
            # Detailed results for each model
            f.write("=" * 70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            for model_name, results in all_results.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * 70 + "\n")
                
                # Classification report
                report = classification_report(
                    results['y_true'], 
                    results['y_pred'],
                    target_names=['Non-Stress', 'Stress'],
                    digits=4
                )
                f.write(report)
                f.write("\n")
                
                # Fold accuracies
                fold_accs = results['fold_accuracies']
                f.write(f"LOSO Fold Accuracies:\n")
                f.write(f"  Mean: {np.mean(fold_accs):.4f}\n")
                f.write(f"  Std:  {np.std(fold_accs):.4f}\n")
                f.write(f"  Min:  {np.min(fold_accs):.4f}\n")
                f.write(f"  Max:  {np.max(fold_accs):.4f}\n")
                f.write("\n")
        
        print(f"Evaluation report saved to {save_path}")


if __name__ == "__main__":
    # Load data
    print("Loading preprocessed data...")
    
    try:
        X = np.load('data/processed/X.npy')
        y = np.load('data/processed/y.npy')
        subject_ids = np.load('data/processed/subject_ids.npy')
    except FileNotFoundError:
        print("Preprocessed data not found. Using sample data...")
        X = np.load('data/processed/X_sample.npy')
        y = np.load('data/processed/y_sample.npy')
        subject_ids = np.load('data/processed/subject_ids_sample.npy')
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    #Model trainer
    trainer = StressDetectionModels()
    
    #Evaluate all models
    all_results = trainer.evaluate_all_models(X, y, subject_ids)
    
    #Plot results
    trainer.plot_results_comparison(all_results)
    
    #Generate report
    trainer.generate_report(all_results)
    
    #Train final model (best performing one)
    best_model_name = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
    final_model, scaler = trainer.train_final_model(X, y, best_model_name)
    
    #Save final model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(final_model, model_dir / 'stress_detector.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    print(f"\nFinal model saved to {model_dir}")
    print("\nTraining complete!")