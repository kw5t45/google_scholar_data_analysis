from analysis import *
import classes as gs
import math
from scholarly import ProxyGenerator
import random

# List of proxies (replace with actual proxies you have access to)
proxies_list = [
    'http://proxy1:port',
    'http://proxy2:port',
    'http://proxy3:port'
]

# Function to get a random proxy from the list
def get_random_proxy():
    return random.choice(proxies_list)

# Setup proxy
proxies = {
    'http': get_random_proxy(),
    'https': get_random_proxy(),
}

# few papers jP1qgO4AAAAJ
# papak id O9d4j7oAAAAJ

inst: gs.Author = gs.Author('O9d4j7oAAAAJ')
print(inst.co_author_ids)
inst.pprint_all_author_data(show_publications=False)
#inst.co_author_ids.remove('O9d4j7oAAAAJ')
for id in inst.co_author_ids:
    x = gs.Author(id)
    x.pprint_all_author_data()
