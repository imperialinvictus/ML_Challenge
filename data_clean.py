import json

import pandas as pd

from config import (
    DEFAULT_DRINK_KEYWORDS,
    DEFAULT_MOVIE_KEYWORDS,
    SPICE_MAP,
    SETTING_COMBINATION_MAP,
    PERSON_COMBINATION_MAP
)
from parsers import (
    get_number_from_response,
    get_movie_vector_from_response,
    get_drink_vector_from_response,
    get_combination_vector
)
from clustering import create_fuzzy_clusters, create_combination_categories


def get_response_numbers():
    data_csv = pd.read_csv("cleaned_data_combined.csv", keep_default_na=False)
    # Columns are
    # id
    # Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)
    # Q2: How many ingredients would you expect this food item to contain?
    # Q3: In what setting would you expect this food to be served? Please check all that apply
    # Q4: How much would you expect to pay for one serving of this food item?
    # Q5: What movie do you think of when thinking of this food item?
    # Q6: What drink would you pair with this food item?
    # Q7: When you think about this food item, who does it remind you of?
    # Q8: How much hot sauce would you add to this food item?
    # Label

    response_dict = {i: dict() for i in range(10)}

    for line in data_csv.values:
        for i in range(10):
            answer = line[i]
            if i in {2}:
                cleaned_answer = get_number_from_response(answer, True)
            elif i in {4}:
                cleaned_answer = get_number_from_response(answer, False)
            elif i in {5}:
                cleaned_answer = get_movie_vector_from_response(answer)
            elif i in {6}:
                cleaned_answer = get_drink_vector_from_response(answer)
            else:
                cleaned_answer = answer
            response_dict[i][cleaned_answer] = response_dict[i].get(cleaned_answer, 0) + 1

    for i in range(10):
        response_dict[i] = dict(sorted(response_dict[i].items(), key=lambda item: item[1], reverse=True))
        print(f"{data_csv.columns[i]}:\n{response_dict[i]}")


def make_cleaned_output_csv(input_csv_filename: str, output_json_filename: str, feature_filename: str):
    data_csv_clean = pd.read_csv(input_csv_filename, keep_default_na=False)
    data_csv_clean.columns = ['id', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q9', 'Label']

    movie_fuzzy_clusters = create_fuzzy_clusters(data_csv_clean['Q5'].tolist(),
                                                 add_keywords=DEFAULT_MOVIE_KEYWORDS, cutoff=85, minimum_size=5)
    drink_fuzzy_clusters = create_fuzzy_clusters(data_csv_clean['Q6'].tolist(),
                                                 add_keywords=DEFAULT_DRINK_KEYWORDS, cutoff=85, minimum_size=5)
    setting_combinations = create_combination_categories(data_csv_clean['Q3'].tolist(), SETTING_COMBINATION_MAP)
    person_combinations = create_combination_categories(data_csv_clean['Q7'].tolist(), PERSON_COMBINATION_MAP)

    for i in range(10):
        match i:
            case 1:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: [int(j == x - 1) for j in range(5)])
            case 2:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_number_from_response(x, True))
            case 3:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_combination_vector(x, setting_combinations, SETTING_COMBINATION_MAP))
            case 4:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_number_from_response(x, False))
            case 5:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_movie_vector_from_response(x, clusters=movie_fuzzy_clusters, cutoff=90))
            case 6:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_drink_vector_from_response(x, clusters=drink_fuzzy_clusters, cutoff=90))
            case 7:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: get_combination_vector(x, person_combinations, PERSON_COMBINATION_MAP))
            case 8:
                data_csv_clean.iloc[:, i] = data_csv_clean.iloc[:, i].apply(
                    lambda x: [int(j == SPICE_MAP[x]) for j in range(5)])

    data_csv_clean.to_json(output_json_filename, orient='records', indent=4)

    feature_guide = {
        'Q1': {
            'type': 'onehot',
            'description': 'Rating of food item from 1 to 5',
            'onehot_labels': list(range(1, 6)),
            'onehot_lengths': [5]
        },
        'Q2': {
            'type': 'number',
            'description': 'How many ingredients in food item'
        },
        'Q3': {
            'type': 'onehot',
            'description': 'Onehot for setting combinations, then individual settings, then count',
            'onehot_labels': list(setting_combinations) + list(SETTING_COMBINATION_MAP) + ['count'],
            'onehot_lengths': [len(list(setting_combinations)), len(list(SETTING_COMBINATION_MAP)), 1]
        },
        'Q4': {
            'type': 'number',
            'description': 'Price range of food item'
        },
        'Q5': {
            'type': 'onehot',
            'description': 'Related movie to food item',
            'onehot_labels': list(movie_fuzzy_clusters.keys()) + ['other'],
            'onehot_lengths': [len(list(movie_fuzzy_clusters.keys())) + 1]
        },
        'Q6': {
            'type': 'onehot',
            'description': 'Related drink to food item',
            'onehot_labels': list(drink_fuzzy_clusters.keys()) + ['other'],
            'onehot_lengths': [len(list(drink_fuzzy_clusters.keys())) + 1]
        },
        'Q7': {
            'type': 'onehot',
            'description': 'Onehot for person combinations, then individual people, then count',
            'onehot_labels': list(person_combinations) + list(PERSON_COMBINATION_MAP) + ['count'],
            'onehot_lengths': [len(list(person_combinations)), len(list(PERSON_COMBINATION_MAP)), 1]
        },
        'Q8': {
            'type': 'onehot',
            'description': 'Spiciness level of food item',
            'onehot_labels': list(SPICE_MAP),
            'onehot_lengths': [len(SPICE_MAP)]
        }
    }

    with open(feature_filename, 'w') as f:
        json.dump(feature_guide, f, indent=4)


# make_cleaned_output_csv('cleaned_data_combined.csv', 'cleaned_output.json', 'feature_guide.json')


def make_cleaned_flattened_dataframe(input_csv_filename: str):
    data_csv_clean = pd.read_csv(input_csv_filename, keep_default_na=False)
    data_csv_clean.columns = ['id', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Label']

    movie_fuzzy_clusters = create_fuzzy_clusters(data_csv_clean['Q5'].tolist(),
                                                 add_keywords=DEFAULT_MOVIE_KEYWORDS, cutoff=85, minimum_size=5)
    drink_fuzzy_clusters = create_fuzzy_clusters(data_csv_clean['Q6'].tolist(),
                                                 add_keywords=DEFAULT_DRINK_KEYWORDS, cutoff=85, minimum_size=5)
    setting_combinations = create_combination_categories(data_csv_clean['Q3'].tolist(), SETTING_COMBINATION_MAP)
    person_combinations = create_combination_categories(data_csv_clean['Q7'].tolist(), PERSON_COMBINATION_MAP)

    # Create a new flattened pandas DataFrame
    processed_data = []
    feature_names = []

    # Process each row
    for idx, row in data_csv_clean.iterrows():
        processed_row = {'id': row['id']}

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
        q5_vector = get_movie_vector_from_response(row['Q5'], clusters=movie_fuzzy_clusters, cutoff=90)
        for i, val in enumerate(q5_vector):
            if i < len(movie_fuzzy_clusters):
                col_name = f'Q5_movie_{list(movie_fuzzy_clusters.keys())[i]}'
            else:
                col_name = 'Q5_movie_other'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        # Process Q6: drink
        q6_vector = get_drink_vector_from_response(row['Q6'], clusters=drink_fuzzy_clusters, cutoff=90)
        for i, val in enumerate(q6_vector):
            if i < len(drink_fuzzy_clusters):
                col_name = f'Q6_drink_{list(drink_fuzzy_clusters.keys())[i]}'
            else:
                col_name = 'Q6_drink_other'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        # Process Q7: person
        q7_vector = get_combination_vector(row['Q7'], person_combinations, PERSON_COMBINATION_MAP)
        for i, val in enumerate(q7_vector[:len(person_combinations)]):
            col_name = f'Q7_person_combo_{i}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        for i, val in enumerate(
                q7_vector[len(person_combinations):len(person_combinations) + len(PERSON_COMBINATION_MAP)]):
            col_name = f'Q7_person_{list(PERSON_COMBINATION_MAP)[i]}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        processed_row['Q7_person_count'] = q7_vector[-1]
        if 'Q7_person_count' not in feature_names:
            feature_names.append('Q7_person_count')

        # Process Q8: spiciness level
        q8_vector = [int(j == SPICE_MAP[row['Q8']]) for j in range(5)]
        for i, val in enumerate(q8_vector):
            col_name = f'Q8_spice_level_{i}'
            processed_row[col_name] = val
            if col_name not in feature_names:
                feature_names.append(col_name)

        processed_row['Label'] = row['Label']
        processed_data.append(processed_row)

    flattened_df = pd.DataFrame(processed_data)

    return flattened_df

df = make_cleaned_flattened_dataframe('cleaned_data_combined.csv')
