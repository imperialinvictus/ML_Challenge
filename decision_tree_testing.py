import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as st

from data_clean import make_cleaned_flattened_dataframe

df = make_cleaned_flattened_dataframe('cleaned_data_combined.csv')

RANDOM_TESTS = 10

# Split the data into X (dependent variables) and t (response variable)
t = np.array(df["Label"])
X = df.drop("Label", axis=1)

X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=300 / 1643, random_state=0)

X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=300 / 1343, random_state=0)

def do_decision_trees():
    print("====== Decision Trees ======")
    tree_res = []
    depths = list(range(1,21))
    for max_depth in depths:
        train_acc = 0
        val_acc = 0
        for i in range(RANDOM_TESTS):
            X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=300 / 1643, random_state=i)

            X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=300 / 1343, random_state=i)

            model = DecisionTreeClassifier(random_state=i, max_depth=max_depth)

            model.fit(X_train, t_train)
            train_acc += 1 - model.score(X_train, t_train)
            val_acc += 1 - model.score(X_valid, t_valid)

        print(f"max_depth={max_depth}", train_acc/RANDOM_TESTS, val_acc/RANDOM_TESTS)
        tree_res.append([train_acc/RANDOM_TESTS, val_acc/RANDOM_TESTS])

    plt.title("Decision Tree")
    plt.plot(depths, [r[0] for r in tree_res], label="train error")
    plt.plot(depths, [r[1] for r in tree_res], label="val error")
    plt.ylabel("error")
    plt.xlabel("max_depth")
    plt.legend()
    plt.show()


def calculate_mode(y_preds):
    mode_labels = []
    for i in range(y_preds.shape[1]):  # Loop through each column (each sample)
        unique, counts = np.unique(y_preds[:, i], return_counts=True)
        mode_labels.append(unique[np.argmax(counts)])  # Select the label with the highest count
    return np.array(mode_labels)


def model_averaging(model, ntries=100, nsample=200):
    """
    Fits model a number of times on a subset of the NHANES data
    set, and returns the average validation error of the
    individual models, along with the validation error of the
    average prediction of the models.

    Parameters:
        `model` - an sklearn model supporting the methods fit(),
                  predict(), and score()
        `ntries` - number of times to train the classifier
        `nsamples` - number of data points to sample to train each
                     classifier

    Returns: A tuple containing the average validation error
             across the individual models, and the validation
             error of the average prediction of the models.
    """

    train_acc = []
    val_acc = []
    ys = []

    for i in range(ntries):
        subset = random.sample(range(len(X_train)), nsample)
        model.fit(X_train.iloc[subset], t_train[subset])
        ys.append(model.predict(X_valid))
        train_acc.append(model.score(X_train.iloc[subset], t_train[subset]))
        val_acc.append(model.score(X_valid, t_valid))

    ys = np.stack(ys)

    # Compute the average validation error across individual models
    avg_val_error = 1 - np.average(val_acc)

    # Calculate mode manually with np.unique
    mode_labels = calculate_mode(ys)

    # Compute the validation error using the majority vote
    ensemble_val_error = 1 - np.average(mode_labels == t_valid)

    return avg_val_error, ensemble_val_error

def do_ensemble():
    print("====== Ensemble Decision Trees ======")
    depths = list(range(1,21))
    for max_depth in depths:
        avg_val_error, ensemble_val_error = 0, 0
        for i in range(RANDOM_TESTS):
            X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=300 / 1643, random_state=i)

            X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=300 / 1343, random_state=i)

            model = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
            add_avg_val_error, add_ensemble_val_error = model_averaging(model, ntries=10)
            avg_val_error += add_avg_val_error
            ensemble_val_error += add_ensemble_val_error
        print(f"max_depth={max_depth}", avg_val_error/RANDOM_TESTS, ensemble_val_error/RANDOM_TESTS)


def estimate_variance(model, ntries=100, nsample=200):
    """
    Estimate the variance of a classifier on the NHANES data set.

    Parameters:
        `model` - an sklearn model supporting the methods fit(),
                  predict(), and score()
        `ntries` - number of times to train the classifier to compute
                   the classifier's variance.
        `nsamples` - number of data points to sample to train each
                     classifier

    Returns: A tuple containing the average training error,
             average validation error, and variance estimate.
    """
    train_acc = []
    val_acc = []
    for i in range(ntries):
        subset = random.sample(range(len(X_train)), nsample)
        model.fit(X_train.iloc[subset], t_train[subset])
        train_acc.append(model.score(X_train.iloc[subset], t_train[subset]))
        val_acc.append(model.score(X_valid, t_valid))
    train_error = 1 - np.average(train_acc)
    val_error = 1 - np.average(val_acc)
    return train_error, val_error

def do_random_forest():
    global X_tv, X_test, t_tv, t_test, X_train, X_valid, t_train, t_valid
    print("====== Random Forest ======")
    forest_res = []
    num_trees = [1, 2, 5, 10, 25, 50]
    for n in num_trees:
        val_error = 0
        train_error = 0
        for i in range(RANDOM_TESTS):
            X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=300 / 1643, random_state=i)

            X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=300 / 1343, random_state=i)

            model = RandomForestClassifier(n_estimators=n)
            add_train_error, add_val_error = estimate_variance(model, ntries=10, nsample=1000)
            print(add_train_error, add_val_error)
            train_error += add_train_error
            val_error += add_val_error
        print(f"num_trees={n}", train_error / RANDOM_TESTS, val_error / RANDOM_TESTS)
        forest_res.append([train_error / RANDOM_TESTS, val_error / RANDOM_TESTS])





n_estimators = [int(x) for x in np.linspace(start = 400, stop = 600, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(30, 50, num = 10)]
min_samples_split = [6, 7, 8, 9, 10]
min_samples_leaf = [1, 2, 3]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model = RandomForestClassifier()
# model_random = RandomizedSearchCV(estimator = model, param_distributions=random_grid, cv=25, random_state=1, n_jobs=-1)

# model_random.fit(X_train, t_train)

# print(model_random.best_params_)


def evaluate(model, test_features, test_labels):
    score = model.score(test_features, test_labels)
    #print('Model Performance')
    #print('Accuracy = {:0.2f}%.'.format(score*100))

    return score

base_acc = 0
for i in range(100):
    base_model = RandomForestClassifier(n_estimators=50, random_state=i)
    base_model.fit(X_train, t_train)
    base_acc += evaluate(base_model, X_valid, t_valid)
base_acc = base_acc / 100
print('Model Performance')
print('Accuracy = {:0.2f}%.'.format(base_acc*100))

random_acc = 0
for i in range(100):
    #best_random = RandomForestClassifier(**model_random.best_params_, random_state=i)
    best_random = RandomForestClassifier(**{'n_estimators': 500, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 40, 'bootstrap': True}, random_state = i)
    best_random.fit(X_train, t_train)
    random_acc += evaluate(best_random, X_valid, t_valid)
random_acc = random_acc / 100
print('Model Performance')
print('Accuracy = {:0.2f}%.'.format(random_acc*100))

print('Improvement of {:0.2f}%.'.format(100 * (random_acc - base_acc) / base_acc))

for i in range(100):
    best_random = RandomForestClassifier(**{'n_estimators': 500, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 40, 'bootstrap': True}, random_state=i)
    best_random.fit(X_train, t_train)
    random_acc += evaluate(best_random, X_test, t_test)
random_acc = random_acc / 100
print('Model Test Performance')
print('Accuracy = {:0.2f}%.'.format(random_acc*100))