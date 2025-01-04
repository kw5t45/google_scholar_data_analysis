import os
import requests
import json
from typing import Dict, List
def get_public_ip():
    try:
        # Use an external service to get the public IP
        response = requests.get("https://api.ipify.org?format=json")
        ip = response.json()["ip"]
        return ip
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


class MaximumRequestsExceeded(Exception):
    pass


def raise_max_requests_exceeded():
    """
    Raises Maximum Requests Exceeded error, printing all tried IP's and appending tried (current) ip to tried ips.txt
    :param ips: List of tried IP addresses.
    :return: None
    """

    current_ip = get_public_ip()

    # Appending IP to ips text, creating it if it doesn't exist already.
    working_dir = os.getcwd()
    file_path = os.path.join(working_dir, "failed_ips.json")

    # case that file doesn't exist, first time that code crashes and json file is created
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([current_ip], f)
            content = current_ip
    else: # case path exists
        with open(file_path, 'r') as f:
            ips = json.load(f)

        ips.append(current_ip)
        with open(file_path, 'w') as f:
            ips = list(set(ips))
            json.dump(ips, f, indent=1)
            content = ips
    # for printing
    content: list | list[str] = ",\n ".join(content) if isinstance(content, list) else content
    raise MaximumRequestsExceeded(fr'Cannot fetch data from Google scholar (request denied - maximum tries exceeded).'
                                  f' Consider changing your IP address and re-trying.\nTried IP addresses:\n {content}.')



def check_dictionary_format(d: Dict[str, Dict[str | int, int]]) -> bool:
    if not isinstance(d, dict):
        return False

    for key, value in d.items():
        # Check if the key is a string
        if not isinstance(key, str):
            return False

        # Check if the value is a dictionary
        if not isinstance(value, dict):
            return False

        # Check if all keys in the nested dictionary are strings and values are integers
        for inner_key, inner_value in value.items():
            if not isinstance(inner_key, str | int) or not isinstance(inner_value, int):
                return False

    return True

