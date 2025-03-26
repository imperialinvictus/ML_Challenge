"""
Functions for clustering and categorizing response data.
"""
import re
from collections import defaultdict
from fuzzup import fuzz


def create_fuzzy_clusters(response_list: list[str], add_keywords=None, cutoff=80, minimum_size=5) -> dict[str, set]:
    cleaned_response_list = [re.sub(r"[^a-z0-9\s]", '', item.strip().lower()).strip() for item in response_list]
    
    fuzzy_clusters = fuzz.fuzzy_cluster(cleaned_response_list, cutoff=cutoff)
    cluster_dict = defaultdict(set)
    count_dict = defaultdict(int)

    # Group words by cluster_id
    for item in fuzzy_clusters:
        cluster_id = item['cluster_id']
        word = item['word']
        cluster_dict[cluster_id].add(word)

    final_clusters = defaultdict(set)
    final_clusters.update(add_keywords)

    # Add cluster categories to added keywords if they overlap
    for cluster_key, word_cluster in cluster_dict.items():
        found = False
        for keyword, keyword_group in add_keywords.items():
            if not keyword_group.isdisjoint(word_cluster):
                final_clusters[keyword].update(word_cluster)
                found = True
        if not found:
            final_clusters[cluster_key].update(word_cluster)

    for cluster_key, word_cluster in final_clusters.items():
        for word in word_cluster:
            count_dict[cluster_key] += cleaned_response_list.count(word)

    # remove clusters that don't have a minimum size
    keys_to_delete = [item for item in final_clusters if count_dict[item] < minimum_size]
    for item in keys_to_delete:
        if item not in add_keywords:
            del final_clusters[item]

    # remove spaces from feature names
    cluster_keys = list(final_clusters.keys())
    for cluster_key in cluster_keys:
        cluster_key_no_space = re.sub(r"\s", '_', cluster_key)
        if cluster_key != cluster_key_no_space:
            final_clusters[cluster_key_no_space] = final_clusters.pop(cluster_key)

    return final_clusters


def create_combination_categories(response_data, combination_map: dict[str, str], minimum_size=5) -> dict[str, int]:
    count_dict = defaultdict(int)
    for line in response_data:
        if line == "None":
            response_split = ''
        else:
            response_split = ''.join([combination_map[i] for i in line.split(',')])
        count_dict[response_split] += 1

    # remove specific combinations that don't appear enough
    combination_dict = {}
    for i, combination in enumerate(count_dict.keys()):
        if count_dict[combination] >= minimum_size:
            combination_dict[combination] = i

    return combination_dict
