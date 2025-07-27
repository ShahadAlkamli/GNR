import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC 
from sklearn.feature_selection import f_classif  # Changed from mutual_info_classif to f_classif
from sklearn.metrics import classification_report, accuracy_score
from math import gamma, log
import tempfile
from multiprocessing import Pool, cpu_count
from scipy.stats import sem
import time
import warnings
import os
from contextlib import contextmanager
from tqdm import tqdm
import scipy.stats as st


# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
POPULATION_SIZE = 500
MAX_GENERATIONS = 30
PATIENCE = 5  # Early stopping if no improvement for 5 generations
ALPHA = 0.05  # Scaling factor for Lévy flight
BETA = 1.8    # Parameter for Lévy distribution
NUM_RUNS = 30  # Number of independent runs for stability and statistical analysis
FILTER_SIZE = 500  # Number of features to select with F-score filter
FIXED_GENES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # Fixed number of genes to select
CR_embed = 0.3  # crossover embed rate
CPR = 0.8       # crossover gene-wise rate



# Dataset Paths
data_files = [
    "/Users/shahadsaeed/Desktop/Colon.arff",
    "/Users/shahadsaeed/Desktop/Lek1.arff",
    "/Users/shahadsaeed/Desktop/Lek2.arff",
    "/Users/shahadsaeed/Desktop/LungM.arff",
    "/Users/shahadsaeed/Desktop/Lym.arff",
    "/Users/shahadsaeed/Desktop/SRBCT.arff"
]

dataset_names = {
    "/Users/shahadsaeed/Desktop/Colon.arff": "Colon",
    "/Users/shahadsaeed/Desktop/Lek1.arff": "Leukemia1",
    "/Users/shahadsaeed/Desktop/Lek2.arff": "Leukemia2",
    "/Users/shahadsaeed/Desktop/LungM.arff": "Lung",
    "/Users/shahadsaeed/Desktop/Lym.arff": "Lymphoma",
    "/Users/shahadsaeed/Desktop/SRBCT.arff": "SRBCT"
}

@contextmanager
def temporary_arff_file(content):
    """Context manager for handling temporary ARFF files safely."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.arff') as temp_file:
            temp_file.writelines(content)
            temp_file.seek(0)
            yield temp_file.name
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def f_score_filter(X, y, k=FILTER_SIZE):
    """Select top k features based on F-score"""
    f_scores, _ = f_classif(X, y)  # Using F-score instead of mutual information
    top_k_indices = np.argsort(f_scores)[-k:]
    return top_k_indices, f_scores

def levy_flight(beta):
    """Generates a Lévy flight step to introduce long-distance jumps for exploration."""
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def load_arff_data(file_path):
    """Load and preprocess ARFF dataset"""
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Ensure unique attribute names and collect feature names
    attr_counts = {}
    feature_names = []
    new_content = []
    for line in content:
        if line.lower().startswith('@attribute'):
            parts = line.split()
            attr_name = parts[1].strip('" ')
            if attr_name in attr_counts:
                attr_counts[attr_name] += 1
                new_attr_name = f"{attr_name}_{attr_counts[attr_name]}"
                line = line.replace(attr_name, new_attr_name, 1)
                feature_names.append(new_attr_name)
            else:
                attr_counts[attr_name] = 0
                feature_names.append(attr_name)
            if '{' in line:
                before, nominals_part = line.split('{', 1)
                nominals, after = nominals_part.split('}', 1)
                cleaned_nominals = ','.join(nom.strip() for nom in nominals.split(','))
                line = f"{before}{{{cleaned_nominals}}}{after}"
        new_content.append(line)

    # Use context manager for safe file handling
    with temporary_arff_file(new_content) as temp_file_path:
        data, meta = arff.loadarff(temp_file_path)

    df = pd.DataFrame(data)

    # Decode categorical values
    for col in df.select_dtypes([object, 'category']):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Impute missing values with mean
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').mean()).values

    # Z-score normalization
    X = StandardScaler().fit_transform(X)

    # Label encoding for the target
    y = LabelEncoder().fit_transform(y)

    return X, y, feature_names[:-1]  # Return features and names (excluding class label)

def evaluate_individual(args):
    """Worker function for parallel evaluation that creates fresh classifier instance"""
    individual, X, y, fixed_n_genes = args
    clf = SVC(kernel='linear', C=1, random_state=42)  # Fresh instance for each evaluation
    
    # Select top N features based on individual weights
    top_n_indices = np.argsort(individual)[-fixed_n_genes:]
    features_selected = X[:, top_n_indices]
    
    if features_selected.shape[1] == 0:
        return 0, 0, 0, 0  # No features selected

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train, test in loo.split(features_selected):
        try:
            clf.fit(features_selected[train], y[train])
            prediction = clf.predict(features_selected[test])[0]
            y_true.append(y[test][0])
            y_pred.append(prediction)
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return 0, 0, 0, 0  # Return zeros on error

    # Compute metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    return accuracy, precision, recall, f1

def nuclear_fission(population, generation, X_best):
    """Performs nuclear fission to create new solutions with more diversity."""
    new_population = []
    population_mean = population.mean(axis=0)
    population_max = population.max(axis=0)
    
    for individual in population:
        sigma = (log(generation + 1) / (generation + 1)) * np.abs(individual - population_mean)
        P_ne_s = np.round(np.random.rand() + 1)
        P_ne_e = np.round(np.random.rand() + 2)
        
        # Calculate Ne_i → choose random individual ≠ current individual
        random_partner = population[np.random.choice(len(population))]
        Ne_i = individual / (random_partner + 1e-8)  # avoid divide by zero

        if np.random.rand() <= 0.5:
            new_sol = np.random.normal(X_best, sigma) + np.random.rand() * (X_best - P_ne_s * Ne_i)
        else:
            new_sol = np.random.normal(individual, sigma) + np.random.rand() * (X_best - P_ne_e * Ne_i)

        mutation_indices = np.random.choice(len(individual), size=int(0.2 * len(individual)), replace=False)
        new_sol[mutation_indices] = 1 - new_sol[mutation_indices]
        new_sol += ALPHA * levy_flight(BETA)
        new_population.append(np.clip(new_sol, 0, 1))

    return np.array(new_population)

def nuclear_fusion(population, X_best, CR_embed=0.3, CPR=0.8):
    """Performs nuclear fusion with optional embedded crossover."""
    new_population = []
    population_size = len(population)
    
    for individual in population:
        partner_indices = np.random.choice(population_size, 2, replace=False)
        partner1, partner2 = population[partner_indices[0]], population[partner_indices[1]]
        
        # Ionization
        if np.random.rand() < 0.5:
            X_ion = partner1 + np.random.rand() * (partner2 - individual)
        else:
            X_ion = partner1 - np.random.rand() * (partner2 - individual)

        if np.allclose(partner1, partner2):
            X_ion += ALPHA * levy_flight(BETA) * (individual - partner1)

        # Embedded crossover decision
        if np.random.rand() < CR_embed:
            # Perform uniform crossover between X_best and random partner
            partner_random = population[np.random.choice(population_size)]
            X_fu = np.where(np.random.rand(len(X_best)) < CPR, X_best, partner_random)
        else:
            # Normal fusion calculation
            if np.random.rand() <= 0.5:
                X_fu = X_ion + np.random.rand() * (partner1 - X_best) + np.random.rand() * (partner2 - X_best)
            else:
                X_fu = X_ion + ALPHA * levy_flight(BETA) * (X_ion - X_best)

        if np.random.rand() < 0.1:
            X_fu = np.random.rand(len(X_best))

        new_population.append(np.clip(X_fu, 0, 1))
    
    return np.array(new_population)


def nro(X, y, fixed_n_genes, feature_names=None, run_num=None, dataset_name=None):
  
    generation_best = []  # Track the best accuracy from each generation

    """Main NRO algorithm with detailed progress tracking"""
    population = np.random.rand(POPULATION_SIZE, X.shape[1])
    best_solution = None
    best_fitness = -np.inf
    accuracy_values = []
    no_improvement_count = 0
    generation_metrics = []

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Run: {run_num+1}/{NUM_RUNS} | Features: {fixed_n_genes}")
    print(f"{'Generation':<12} {'Best':<10} {'Average':<10} {'Worst':<10} {'Stall':<10}")
    print("-" * 60)

    for generation in range(MAX_GENERATIONS):
        # Prepare arguments for parallel evaluation
        args = [(individual, X, y, fixed_n_genes) for individual in population]
        
        with Pool(cpu_count()) as pool:
            results = pool.map(evaluate_individual, args)

        fitness_scores = [r[0] for r in results]
        accuracy_values.extend(fitness_scores)

        best_idx = np.argmax(fitness_scores)
        current_best = fitness_scores[best_idx]
        generation_best.append(current_best)
        avg_fitness = np.mean(fitness_scores)
        worst_fitness = min(fitness_scores)

        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[best_idx]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Store generation metrics
        generation_metrics.append({
            'generation': generation + 1,
            'best': current_best,
            'average': avg_fitness,
            'worst': worst_fitness,
            'stall': no_improvement_count
        })

        # Print current generation stats
        print(f"{generation+1:<12} {current_best:<10.4f} {avg_fitness:<10.4f} {worst_fitness:<10.4f} {no_improvement_count:<10}")

        if no_improvement_count >= PATIENCE:
            print(f"\nEarly stopping triggered at generation {generation+1}")
            print(f"Best fitness stalled at {best_fitness:.4f} for {PATIENCE} generations")
            break


        population = nuclear_fission(population, generation, X_best=best_solution)
        population = nuclear_fusion(population, best_solution)


    # Get the best metrics from the final evaluation
    args = [(best_solution, X, y, fixed_n_genes)]
    with Pool(1) as pool:
        final_results = pool.map(evaluate_individual, args)
    
    best_accuracy, best_precision, best_recall, best_f1 = final_results[0]

    # Print run summary
    print(f"\nRun completed: {generation+1} generations")
    print(f"Final Best Accuracy: {best_accuracy:.4f}")
    print(f"Final Best Precision: {best_precision:.4f}")
    print(f"Final Best Recall: {best_recall:.4f}")
    print(f"Final Best F1-Score: {best_f1:.4f}")
    print(f"Selected Features: {', '.join(feature_names[i] for i in np.argsort(best_solution)[-fixed_n_genes:][:3])}...")
    print("="*80)
    
    top_n_indices = np.argsort(best_solution)[-fixed_n_genes:]
    selected_features = [feature_names[i] for i in top_n_indices] if feature_names else top_n_indices
    
    return {
        'solution': best_solution,
        'worst_acc': min(generation_best),
        'best_acc': best_accuracy,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_f1': best_f1,
        'avg_acc': np.mean(generation_best),
        'selected_features': selected_features,
        'selected_indices': top_n_indices,
        'generation_metrics': generation_metrics,
        'stopped_early': no_improvement_count >= PATIENCE,
        'stopped_at': generation + 1 if no_improvement_count >= PATIENCE else MAX_GENERATIONS
    }

if __name__ == "__main__":
    results = []
    full_results = []
    detailed_metrics = []
    
    print("\nNuclear Reaction Optimization for Feature Selection")
    print("="*80)
    print(f"{'Dataset':<12} {'Total Genes':<12} {'Filtered Genes':<15} {'Fixed Genes':<15} {'Best Acc':<10} {'Best Prec':<10} {'Best Rec':<10} {'Best F1':<10} {'Worst Acc':<10} {'Avg Acc':<10} {'Top Features':<30} {'Time (s)':<10} {'Stopped Early':<15}")
    print("-" * 180)

    for file_path in data_files:
        dataset_name = dataset_names[file_path]
        print(f"\n{'='*80}")
        print(f"STARTING DATASET: {dataset_name}")
        print("Loading and preprocessing data...")
        
        X, y, all_feature_names = load_arff_data(file_path)
        total_genes = X.shape[1]
        
        print("Applying F-score filter...")
        top_k_indices, _ = f_score_filter(X, y)  # Changed to F-score filter
        X_filtered = X[:, top_k_indices]
        filtered_feature_names = [all_feature_names[i] for i in top_k_indices]

        for fixed_genes in FIXED_GENES:
            print(f"\n{'='*80}")
            print(f"STARTING FEATURE SET SIZE: {fixed_genes}")
            
            best_accuracies = []
            best_precisions = []
            best_recalls = []
            best_f1s = []
            worst_accuracies = []
            avg_accuracies = []
            execution_times = []
            best_features = None
            early_stops = 0

            for run in range(NUM_RUNS):
                start_time = time.time()
                
                result = nro(
                    X_filtered, 
                    y, 
                    fixed_genes, 
                    feature_names=filtered_feature_names, 
                    run_num=run,
                    dataset_name=dataset_name
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                if best_features is None or result['best_acc'] > max(best_accuracies, default=0):
                    best_features = result['selected_features']
                
                best_accuracies.append(result['best_acc'])
                best_precisions.append(result['best_precision'])
                best_recalls.append(result['best_recall'])
                best_f1s.append(result['best_f1'])
                worst_accuracies.append(result['worst_acc'])
                avg_accuracies.append(result['avg_acc'])
                
                if result['stopped_early']:
                    early_stops += 1

                # Store detailed metrics
                for metric in result['generation_metrics']:
                    detailed_metrics.append({
                        'dataset': dataset_name,
                        'fixed_genes': fixed_genes,
                        'run': run+1,
                        'generation': metric['generation'],
                        'best_acc': metric['best'],
                        'avg_acc': metric['average'],
                        'worst_acc': metric['worst'],
                        'stall_count': metric['stall'],
                        'stopped_early': result['stopped_early'],
                        'stopped_at': result['stopped_at']
                    })

                full_results.append({
                    'dataset': dataset_name,
                    'fixed_genes': fixed_genes,
                    'run': run+1,
                    'best_acc': result['best_acc'],
                    'best_precision': result['best_precision'],
                    'best_recall': result['best_recall'],
                    'best_f1': result['best_f1'],
                    'worst_acc': result['worst_acc'],
                    'avg_acc': result['avg_acc'],
                    'selected_features': ', '.join(result['selected_features']),
                    'time': execution_time,
                    'stopped_early': result['stopped_early'],
                    'stopped_at': result['stopped_at'],
                    'generations': result['stopped_at']
                })

            # Calculate statistics
            avg_time = np.mean(execution_times)
            best_avg_acc = np.mean(best_accuracies)
            ci_low, ci_high = st.t.interval(
               0.95, len(best_accuracies) - 1,
               loc=best_avg_acc,
               scale=st.sem(best_accuracies)
            )
            ci_lower = ci_low
            ci_upper = ci_high
            
            
            # Prepare feature list for display (show first 3 features)
            features_display = ', '.join(best_features[:3])
            if len(best_features) > 3:
                features_display += ', ...'
            
            print(f"\nCompleted {NUM_RUNS} runs for {fixed_genes} features")
            print(f"Early stops: {early_stops}/{NUM_RUNS} ({early_stops/NUM_RUNS:.1%})")
            print(f"Best Accuracy: {100*np.max(best_accuracies):.2f}%")
            print(f"Best Precision: {100*np.max(best_precisions):.2f}%")
            print(f"Best Recall: {100*np.max(best_recalls):.2f}%")
            print(f"Best F1-Score: {100*np.max(best_f1s):.2f}%")
            print(f"Worst Accuracy: {100*np.min(worst_accuracies):.2f}%")
            print(f"Average Accuracy: {100*best_avg_acc:.2f}%")
            print(f"Top Features: {features_display}")
            print(f"Average Time: {avg_time:.2f}s")
            print("="*80)

            results.append([
                dataset_name, 
                total_genes, 
                FILTER_SIZE, 
                fixed_genes,
                100*np.max(best_accuracies),
                100*np.max(best_precisions),
                100*np.max(best_recalls),
                100*np.max(best_f1s),
                100*np.min(worst_accuracies),
                100*best_avg_acc,
                ', '.join(best_features),  # Full feature list
                avg_time,
                f"[{100*ci_lower:.2f}%, {100*ci_upper:.2f}%]",
                f"{early_stops}/{NUM_RUNS}"
            ])

    # Save results
    results_df = pd.DataFrame(results, columns=[
        "Dataset", "Total Genes", "Filtered Genes", "Fixed Genes", 
        "Best Acc (%)", "Best Prec (%)", "Best Rec (%)", "Best F1 (%)",
        "Worst Acc (%)", "Avg Acc (%)", 
        "Selected Features", "Time (s)", "CI (95%)", "Early Stops"
    ])
    results_df.to_csv("nro_svm_final_results.csv", index=False)
    
    full_results_df = pd.DataFrame(full_results)
    full_results_df.to_csv("nro_svm_full_run_details.csv", index=False)
    
    detailed_metrics_df = pd.DataFrame(detailed_metrics)
    detailed_metrics_df.to_csv("nro_svm_generation_metrics.csv", index=False)

    print("\nFinal Results Summary:")
    print(results_df.to_string(index=False))
    print("\nResults saved to:")
    print("- nro_svm_final_results.csv (summary)")
    print("- nro_svm_full_run_details.csv (per-run details)")
    print("- nro_svm_generation_metrics.csv (per-generation metrics)")