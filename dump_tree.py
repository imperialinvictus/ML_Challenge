import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from tqdm.notebook import tqdm
from text_cluster.data_clean import get_dataframe_from_file, save_clusters_to_file, get_dataframe_from_csv
from text_cluster.standalone_data_clean import get_dataframe_from_file as get_dataframe_from_file_standalone

from sklearn.tree import _tree

best_params = {
    "n_estimators": 250,
    "max_features": 25,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "oob_score": True
}

best_model = RandomForestClassifier(**best_params, random_state=42)
data_train, data_valid, data_test = save_clusters_to_file('cleaned_data_combined.csv', 'text_cluster', cutoff=85, minimum_size=5, random_state=42)
df_train, df_valid = get_dataframe_from_csv(data_train, 'text_cluster', fuzzy_cutoff=85), get_dataframe_from_csv(data_valid, 'text_cluster', fuzzy_cutoff=85)

t_train, t_valid = np.array(df_train["Label"]), np.array(df_valid["Label"])
X_train, X_valid = df_train.drop("Label", axis=1), df_valid.drop("Label", axis=1)

best_model.fit(X_train, t_train)
print(best_model.score(X_valid, t_valid))

feature_importance = sorted(list(zip([float(i) for i in best_model.feature_importances_], best_model.feature_names_in_)), reverse=True)
for feature in feature_importance:
    print(f"{feature[1]}: {feature[0]:3f}")

def tree_to_code(tree, feature_names, output_py="randomforest_dump.py", output_func='tree', mode='a'):
    """
    From https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    """
    with open(output_py, mode) as f:
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        f.write("def {}({}):\n".format(output_func, ", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                f.write("{}if {} <= {}:\n".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                f.write("{}else:  # if {} > {}\n".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                f.write("{}return {}\n".format(indent, [float(i) for i in tree_.value[node][0]]))

        recurse(0, 1)

for i in range(len(best_model.estimators_)):
    tree_to_code(best_model.estimators_[i], best_model.feature_names_in_, output_func=f"tree{i}", mode='a')