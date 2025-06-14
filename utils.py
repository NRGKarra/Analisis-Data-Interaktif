import pandas as pd
import numpy as np

def create_sample_datasets():
    np.random.seed(42)
    n_samples = 200
    ages = np.random.randint(15, 75, n_samples)
    sexes = np.random.choice(['M', 'F'], n_samples)
    bps = np.random.choice(['LOW', 'NORMAL', 'HIGH'], n_samples)
    cholesterols = np.random.choice(['NORMAL', 'HIGH'], n_samples)
    na_to_k = np.random.uniform(6, 40, n_samples)
    drugs = []
    for i in range(n_samples):
        if bps[i] == 'HIGH' and cholesterols[i] == 'HIGH':
            if ages[i] > 50:
                drugs.append('DrugY')
            else:
                drugs.append('drugA')
        elif bps[i] == 'LOW':
            if cholesterols[i] == 'HIGH':
                drugs.append('drugC')
            else:
                drugs.append('drugX')
        else:
            if na_to_k[i] > 15:
                drugs.append('DrugY')
            else:
                drugs.append('drugX')
    drug_data = pd.DataFrame({
        'Age': ages,
        'Sex': sexes,
        'BP': bps,
        'Cholesterol': cholesterols,
        'Na_to_K': na_to_k,
        'Drug': drugs
    })
    mushroom_features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                        'stalk-shape', 'stalk-root', 'ring-type', 'spore-print-color',
                        'population', 'habitat']
    mushroom_data = pd.DataFrame()
    for feature in mushroom_features:
        if feature == 'odor':
            values = np.random.choice(['a', 'l', 'c', 'y', 'f', 'n', 's', 'p'], n_samples)
        else:
            values = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x', 'y'], n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        mushroom_data[feature] = values
    classes = []
    for odor in mushroom_data['odor']:
        if odor in ['f', 'y', 's', 'c']:
            classes.append('p')
        else:
            classes.append('e')
    mushroom_data['class'] = classes
    return {
        'drug': drug_data,
        'mushroom': mushroom_data
    }

def export_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return filename

def generate_model_report(model_results, task_type):
    report = {
        'task_type': task_type,
        'models_compared': list(model_results.keys()),
        'best_model': None,
        'metrics': model_results
    }
    if task_type == 'classification':
        best_model = max(model_results.keys(), key=lambda x: model_results[x].get('accuracy', 0))
        report['best_model'] = best_model
        report['best_metric'] = 'accuracy'
    else:
        best_model = max(model_results.keys(), key=lambda x: model_results[x].get('r2_score', -float('inf')))
        report['best_model'] = best_model
        report['best_metric'] = 'r2_score'
    return report
