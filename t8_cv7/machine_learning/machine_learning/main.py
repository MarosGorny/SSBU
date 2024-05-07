from sklearn.linear_model import LogisticRegression
from sklearn import svm

import sys
sys.path.insert(0, 'D:/School/Ing/2. Semester/SSBU/SSBULinda/SSBU/t8_cv7/machine_learning/machine_learning')

from data_handling import Dataset
from result_plots import Plotter
from experiment import Experiment



def main():
    """
    Main function to execute the model training and evaluation pipeline.

    Initializes the dataset, defines models and their parameter grids,
    and invokes the replication of model training and evaluation.
    """
    # Initialize dataset and preprocess the data
    dataset = Dataset()

    

    # Define models to be trained
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear'),  # Solver specified for clarity

    }

    # Define hyperparameter grids for tuning
    param_grids = {
        "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [10000]},
    }

    # Add Support Vector Machine model and hyperparameter grid
    models["SVM"] = svm.SVC()
    param_grids["SVM"] = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale']}

    experiment = Experiment(models, param_grids, n_replications=10)
    results = experiment.run(dataset.data, dataset.target)

    # Plot the results using the Plotter class
    plotter = Plotter()
    plotter.plot_metric_density(results)
    plotter.plot_evaluation_metric_over_replications(
        experiment.results.groupby('model')['accuracy'].apply(list).to_dict(),
        'Accuracy per Replication and Average Accuracy', 'Accuracy')
    plotter.plot_confusion_matrices(experiment.mean_conf_matrices)
    plotter.plot_precision_over_replications(
        experiment.results.groupby('model')['precision'].apply(list).to_dict(),
        'Precision per Replication and Average Precision', 'Precision')
    plotter.print_best_parameters(results)


if __name__ == "__main__":
    main()
