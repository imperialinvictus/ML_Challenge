"""
Configuration constants for data cleaning
"""
from collections import defaultdict

W2N_FUZZY_THRESHOLD = 100

DEFAULT_DRINK_KEYWORDS = defaultdict(set)
DEFAULT_DRINK_KEYWORDS.update({
    'none': {'none', 'n/a', 'no', 'na', 'nan', '-', 'not applicable', 'idk', 'nothing', 'no drink'},
    'coke': {'coke', 'cola', 'coca cola', 'cocacola', 'coka-cola'},
    'pepsi': {'pepsi'},
    'pop': {'pop', 'soda', 'sodapop', 'popsoda', 'pops', 'soft drink', 'pop drink'},
    'sparkling': {'sparkling water'},
    'water': {'water'},
    'ice_tea': {'iced tea', 'ice tea'},
    'bubble_tea': {'bubble tea'},
    'green_tea': {'green tea'},
    'tea': {'tea'},
    'milk': {'milk'},
    'juice': {'juice'},
    'canada': {'canada dry'},
    'ginger ale': {'ginger ale', 'gingerale'},
    'beer': {'beer', 'rootbeer'},
    'wine': {'wine'}
})

DEFAULT_MOVIE_KEYWORDS = defaultdict(set)
DEFAULT_MOVIE_KEYWORDS.update({
    'none': {'none', 'n/a', 'no', 'na', 'nan', '-', 'not applicable', 'idk', 'no movie', 'nothing'}
})

SPICE_MAP = {
    'None': 0,
    'A little (mild)': 1,
    'A moderate amount (medium)': 2,
    'A lot (hot)': 3,
    'I will have some of this food item with my hot sauce': 4
}

SETTING_COMBINATION_MAP = {
    'Week day lunch': "0",
    'Week day dinner': "1",
    'Weekend lunch': "2",
    'Weekend dinner': "3",
    'At a party': "4",
    'Late night snack': "5"
}

PERSON_COMBINATION_MAP = {
    'Parents': "0",
    'Siblings': "1",
    'Friends': "2",
    'Teachers': "3",
    'Strangers': "4"
}

DEFAULT_INPUT_CSV = "cleaned_data_combined.csv"
DEFAULT_OUTPUT_JSON = "cleaned_output.json"
DEFAULT_FEATURE_GUIDE = "feature_guide.json"
