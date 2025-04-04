"""
Functions for parsing survey responses
"""
import re
import numpy as np
# from thefuzz import fuzz as fuzzymatch
from collections import defaultdict

WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}


def levenshtein(str1, str2):
    """
    Computes the Levenshtein distance between two strings.

    Source: GeeksForGeeks
    URL: https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/
    """
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]


def fuzzymatch(s1, s2):
    max_len = max(len(s1), len(s2), 1)
    return (1 - (levenshtein(s1, s2) / max_len))*100


def clean_response(response: str) -> str:
    return ''.join([i if ord(i) < 128 else '-' for i in str(response).lower().strip()])  # remove non-ascii


def get_num_from_element(element: str) -> float | None:
    try:
        return float(element)
    except ValueError:
        fuzz_element = re.sub(r'[^a-z]', '', element)
        if fuzz_element in WORD2NUM:
            return WORD2NUM[fuzz_element]
    return None


def get_number_from_response(response: str, item_list: bool) -> float | None:
    cleaned = clean_response(response)

    punctuation_list = r"[!\"#$%&'()*+,\/:;<=>?@\[\\\]^_`{|}~]|\s"
    cleaned_list = re.split(punctuation_list, cleaned)
    cleaned_list = [i for i in cleaned_list if i.strip()]

    final_list = []
    for element in cleaned_list:
        if element in {'-', 'to', 'and'}:
            final_list.append('-')
            continue
        for i in element.split("-"):
            final_list.append(i)
            final_list.append('-')
        final_list.pop(-1)  # remove ending dash

    first_num = False
    detect_range = [False, False]
    for element in final_list:
        if element == '-':
            if isinstance(detect_range[0], float):
                detect_range[1] = True
            else:
                detect_range = [False, False]
            continue
        number = get_num_from_element(element)
        if number is None:
            detect_range = [False, False]
            continue
        if first_num is False:
            first_num = number
        if detect_range[1]:
            return (detect_range[0] + number) / 2  # This is average of the first range in the string
        if detect_range[0] is False:
            detect_range[0] = number

    if first_num is not False:
        return first_num  # There is no decipherable range in the string, take first number
    if not item_list:
        return 10  # There are no decipherable numbers in the string
    else:  # might be ingredient list
        commas = cleaned.count(",")
        if commas > 0:
            return commas + 1
        else:
            return len(re.findall(r"\n+", cleaned)) + 1


def get_number_from_response_2(value: str, item_list: bool) -> float | None:
    """Convert string to numeric value"""
    if not value or (isinstance(value, float) and np.isnan(value)):
        return 0.0

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # Map word numbers to digits
    # fmt: off
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50
    }
    # fmt: on

    # Convert to lowercase for word matching
    text = value.lower()

    # Extract numbers (both digits and words)
    numbers = []

    # Check for word numbers
    for word, num in word_to_num.items():
        if word in text:
            numbers.append(num)

    # Check for digit numbers using regex
    digit_matches = re.findall(r"\d+", text)
    for match in digit_matches:
        numbers.append(int(match))

    # If we found any numbers, return the average
    if numbers:
        return sum(numbers) / len(numbers)
    
    if not item_list:
        return 10  # There are no decipherable numbers in the string
    else:  # might be ingredient list
        commas = text.count(",")
        if commas > 0:
            return commas + 1
        else:
            return len(re.findall(r"\n+", text)) + 1


def get_movie_vector_from_response(response: str, cutoff=90, clusters=None) -> list[int]:
    cleaned = clean_response(response)
    movie = re.sub(r"[^a-z0-9\s]", '', cleaned).strip()

    keywords = defaultdict(set)
    keywords.update(clusters)

    bools = {}
    for key, values in keywords.items():
        bools[key] = 0
        for kw in values:
            if fuzzymatch(kw, movie) > cutoff:
                bools[key] = 1
                break  # Stop checking

    # Set 'other' if none of the other options match
    bools['other'] = int(not any(bools.values()))

    return list(bools.values())


def get_drink_vector_from_response(response: str, cutoff=90, clusters=None) -> list[int]:
    cleaned = clean_response(response)
    drink = re.sub(r"[^a-z0-9\s]", '', cleaned).strip()

    keywords = defaultdict(set)
    keywords.update(clusters)

    bools = {key: int(any(kw in drink for kw in values)) for key, values in keywords.items()}

    # Exclude subcategories
    if bools['ice_tea'] or bools['bubble_tea'] or bools['green_tea']:
        bools['tea'] = 0
    if bools['sparkling']:
        bools['water'] = 0

    # Set 'other' if none of the other options match
    bools['other'] = int(not any(bools.values()))

    return list(bools.values())


def get_combination_vector(response: str, combination_categories: dict[str, int], combination_map: dict[str, str]) -> \
        list[int]:
    if response == "None":
        response_split = ''
    else:
        response_split = ''.join([combination_map[i] for i in response.split(',')])

    # one-hots for each combination, with other at the end
    combo_bools = [int(i == combination_categories.get(response_split, False)) for i in
                   range(len(combination_categories))]
    combo_bools.append(int(not any(combo_bools)))
    # one-hots for each individual response
    indiv_bools = [int(str(i) in response_split) for i in range(5)]
    # count of number of chosen
    count = len(response_split)

    return combo_bools + indiv_bools + [count]
