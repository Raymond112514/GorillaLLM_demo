import json
import os
import re
import statistics
from pathlib import Path
from typing import Union

import numpy as np
from constant import *

from tqdm import tqdm


def is_multi_turn(test_category):
    return "multi_turn" in test_category

def contain_multi_turn_irrelevance(test_category):
    return "miss_func" in test_category or "miss_param" in test_category

def is_executable(test_category):
    return "exec" in test_category or "rest" in test_category


def is_rest(test_category):
    return "rest" in test_category


def is_relevance_or_irrelevance(test_category):
    return "relevance" in test_category or "irrelevance" in test_category


def is_chatable(test_category):
    return "chatable" in test_category


def is_java(test_category):
    return "java" in test_category


def is_js(test_category):
    return "javascript" in test_category


def is_sql(test_category):
    return "sql" in test_category


def load_file(file_path):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result


def get_handler(model_name):
    return handler_map[model_name](model_name, temperature=0)  #Temperature doesn't matter for evaluation


def write_list_of_dicts_to_file(filename, data, subdir=None):
    if subdir:
        # Ensure the subdirectory exists
        os.makedirs(subdir, exist_ok=True)

        # Construct the full path to the file
        filename = os.path.join(subdir, filename)

    # Write the list of dictionaries to the file in JSON format
    with open(filename, "w") as f:
        for i, entry in enumerate(data):
            # Go through each key-value pair in the dictionary to make sure the values are JSON serializable
            for key, value in entry.items():
                try:
                    json.dumps(value)
                except:
                    # If the value is not JSON serializable, wrap it in a string
                    entry[key] = str(value)

            json_str = json.dumps(entry)
            f.write(json_str)
            if i < len(data) - 1:
                f.write("\n")


def is_function_calling_format_output(decoded_output):
    # Ensure the output is a list of dictionaries
    if type(decoded_output) == list:
        for item in decoded_output:
            if type(item) != dict:
                return False
        return True
    return False


def is_executable_format_output(decoded_output):
    # Ensure the output is a list of strings (one or more strings)
    if type(decoded_output) == list:
        if len(decoded_output) == 0:
            return False
        for item in decoded_output:
            if type(item) != str:
                return False
        return True
    return False


def is_rest_format_output(decoded_output):
    # Ensure the output is a list of one string
    if type(decoded_output) == list:
        if len(decoded_output) == 1 and type(decoded_output[0]) == str:
            return True
    return False


def is_empty_output(decoded_output):
    # This function is a patch to the ast decoder for relevance detection
    # Sometimes the ast decoder will parse successfully, but the input doens't really have a function call
    # [], [{}], and anything that is not in function calling format is considered empty (and thus should be marked as correct)
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False

