import pandas as pd
from word2num import Word2Num
import regex as re

w2n = Word2Num(fuzzy_threshold=100)


def get_num_from_element(element: str) -> float | None:
	try:
		return float(element)
	except ValueError:
		if len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", element.lower())) > 0:
			return w2n.parse(element)
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
		final_list.pop(-1)

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
		return first_num  # There is no decipherable range in the string
	if not item_list:
		return None  # There are no decipherable numbers in the string
	else:  # we can just look for lists of items
		commas = cleaned.count(",")
		if commas > 0:
			return commas + 1
		else:
			return len(re.findall(r"\n+", cleaned)) + 1


def clean_response(response: str) -> str:
	return ''.join([i if ord(i) < 128 else '-' for i in response.lower().strip()])  # remove non-ascii


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
		elif i in {5, 6}:
			cleaned_answer = clean_response(answer)
		else:
			cleaned_answer = answer
		response_dict[i][cleaned_answer] = response_dict[i].get(cleaned_answer, 0) + 1

for i in range(10):
	response_dict[i] = dict(sorted(response_dict[i].items(), key=lambda item: item[1], reverse=True))
	print(f"{data_csv.columns[i]}:\n{response_dict[i]}")
