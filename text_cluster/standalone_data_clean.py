import json

import pandas as pd

from text_cluster.config import (
    DEFAULT_DRINK_KEYWORDS,
    DEFAULT_MOVIE_KEYWORDS,
    SPICE_MAP,
    SETTING_COMBINATION_MAP,
    PERSON_COMBINATION_MAP
)
from text_cluster.parsers import (
    get_number_from_response,
    get_movie_vector_from_response,
    get_drink_vector_from_response,
    get_combination_vector
)


def make_cleaned_flattened_dataframe_from_input_file(input_csv_filename: str, movie_fuzzy_clusters, drink_fuzzy_clusters,
                                               setting_combinations, person_combinations, has_labels=False, fuzzy_cuttoff=90):
    data_csv_clean = pd.read_csv(input_csv_filename, keep_default_na=False)
    csv_columns = ['id', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'] + (['Label'] if has_labels else [])
    data_csv_clean.columns = csv_columns

    # Create a new flattened pandas DataFrame
    processed_data = []
    feature_names = []

    # Process each row
    for idx, row in data_csv_clean.iterrows():
        processed_row = {}

        # Process Q1: complexity rating (1-5)
        q1_vector = [int(j == row['Q1'] - 1) for j in range(5)]
        for i, val in enumerate(q1_vector):
            col_name = f'Q1_complexity_{i + 1}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        # Process Q2: number of ingredients
        processed_row['Q2_ingredients'] = get_number_from_response(row['Q2'], True)
        if 'Q2_ingredients' not in feature_names:
            feature_names.append('Q2_ingredients')

        # Process Q3: setting combinations
        q3_vector = get_combination_vector(row['Q3'], setting_combinations, SETTING_COMBINATION_MAP)
        for i, val in enumerate(q3_vector[:len(setting_combinations)]):
            col_name = f'Q3_setting_combo_{i}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        for i, val in enumerate(
                q3_vector[len(setting_combinations):len(setting_combinations) + len(SETTING_COMBINATION_MAP)]):
            col_name = f'Q3_setting_{list(SETTING_COMBINATION_MAP)[i]}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        processed_row['Q3_setting_count'] = q3_vector[-1]
        if 'Q3_setting_count' not in feature_names:
            feature_names.append('Q3_setting_count')

        # Process Q4: price
        processed_row['Q4_price'] = get_number_from_response(row['Q4'], False)
        if 'Q4_price' not in feature_names:
            feature_names.append('Q4_price')

        # Process Q5: movie
        q5_vector = get_movie_vector_from_response(row['Q5'], clusters=movie_fuzzy_clusters, cutoff=fuzzy_cuttoff)
        for i, val in enumerate(q5_vector):
            if i < len(movie_fuzzy_clusters):
                col_name = f'Q5_movie_{list(movie_fuzzy_clusters.keys())[i]}'
            else:
                col_name = 'Q5_movie_other'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        # Process Q6: drink
        q6_vector = get_drink_vector_from_response(row['Q6'], clusters=drink_fuzzy_clusters, cutoff=fuzzy_cuttoff)
        for i, val in enumerate(q6_vector):
            if i < len(drink_fuzzy_clusters):
                col_name = f'Q6_drink_{list(drink_fuzzy_clusters.keys())[i]}'
            else:
                col_name = 'Q6_drink_other'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        # Process Q8: spiciness level
        q8_vector = [int(j == SPICE_MAP[row['Q8']]) for j in range(5)]
        for i, val in enumerate(q8_vector):
            col_name = f'Q8_spice_level_{i}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        if has_labels:
            processed_row['Label'] = row['Label']

        processed_data.append(processed_row)

    flattened_df = pd.DataFrame(processed_data)

    return flattened_df


def save_dict_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump({k: list(v) if isinstance(v, set) else v for k, v in data.items()}, f)


def load_dict_from_json(filename):
    with open(filename, "r") as f:
        loaded = json.load(f)
        return {k: set(v) if isinstance(v, list) else v for k, v in loaded.items()}


def get_dataframe_from_file(filename, folder_path, fuzzy_cutoff=90, has_labels=False):
    movie_fuzzy_clusters = load_dict_from_json(f'{folder_path}/movie_clusters.json')
    drink_fuzzy_clusters = load_dict_from_json(f'{folder_path}/drink_clusters.json')
    setting_combinations = load_dict_from_json(f'{folder_path}/setting_combinations.json')
    person_combinations = load_dict_from_json(f'{folder_path}/person_combinations.json')

    df_file = make_cleaned_flattened_dataframe_from_input_file(filename, movie_fuzzy_clusters, drink_fuzzy_clusters,
                                                         setting_combinations, person_combinations, fuzzy_cuttoff=fuzzy_cutoff, has_labels=has_labels)
    return df_file
