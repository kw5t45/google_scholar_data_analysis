from scholarly import scholarly
import json
from tqdm import tqdm
import sys
import os
import sympy as sp
import math
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os


### TO FIX
# BUG WITH PADDING CITATIONS PER YEAR DATA 1945 IN TAO WTF
# STRINGS AS YEARS IN DICTIONARIES SHOULD BE INTEGERS
# SOME CO AUTHORS DONT APPEAR
# PROXIES
# WIEGHT CITATIONS SHOULD RETURN FLOAT NOT INT
# MAKE FUNCTIONS NORMAL FUNCTIONS OUTSIDE OF CLASSES

def get_citations_per_year_per_paper(json_name) -> dict:
    '''

    :param json_name: json file containing authors paper data
    :return: dictionary in form:
         {'paper1': {'2011': 5, '2012': 3},
          'paper2': {'2010': 2, '2013': 7}...}
          where nested dictionary is the citations per year
    '''
    paper_data: list = get_paper_data_from_json(json_name)  # list of all papers
    dic = {}
    for paper in paper_data:
        dic[paper[0]] = paper[-1]

    return dic


def get_paper_params(scholar_id):
    '''

    :param scholar_id:
    :return: (papers)  [title, pub_year, citation, author_pub_id, num_citations, cited_by_url,
                                 cited_id, pages, publisher, cites_id, cites_per_year]
    '''
    author: dict = scholarly.search_author_id(scholar_id)
    name: str = author['name']
    search_query = scholarly.search_author(name)
    author = scholarly.fill(next(search_query))  # gets more data by searching by name
    # cleaning Author's publication data

    publications = []
    for index, value in tqdm(enumerate(author['publications']), desc=f'Getting author {scholar_id} paper data:',
                             total=len(author['publications'])):
        current_iterable_publication = scholarly.fill(author['publications'][index])
        # progress bar updating
        if index % 10 == 0:
            # Clear the line for visual effect
            sys.stdout.write("\033[K")
            # Re-display the progress bar
        #####
        # we keep title, pub_year, citation, author_pub_id, num_citations,
        # cited_by_url,
        # cited_id, pages, publisher, cites_id, cites_per_year
        # for each paper
        #####
        publication_data = []
        title = author['publications'][index]['bib']['title']
        if title == 'REVIEWERS LIST A':  # vgazei san teleytaio paper kati periergo gia ayto to kanw hardcode
            break
        current_paper_authors: str = current_iterable_publication['bib']['author']
        current_paper_authors: list = current_paper_authors.split(" and ")

        pub_year = current_iterable_publication['bib']['pub_year']
        citation = current_iterable_publication['bib']['citation']
        author_pub_id = current_iterable_publication['author_pub_id']
        num_citations = current_iterable_publication['num_citations']
        try:
            pages = current_iterable_publication['bib']['pages']
        except KeyError:
            pages = 'NULL'
        try:
            publisher = current_iterable_publication['bib']['publisher']
        except KeyError:
            publisher = 'NULL'
        try:
            cited_by_url = current_iterable_publication['citedby_url']
        except KeyError:
            cited_by_url = 'NULL'
        try:
            cites_per_year = current_iterable_publication['cites_per_year']
        except KeyError:
            cites_per_year = ['NULL']

        # padding 0 values in years that got no citations, from publication year up to current year

        for year in range(pub_year, 2024 + 1):
            if year not in cites_per_year:
                cites_per_year[str(year)] = 0

        try:
            cited_id = current_iterable_publication['cites_id'][0]  # is list for some reason

        except KeyError:  # some publications have no cites id or url for some reason
            cited_by_url = 'NULL'
            cited_id = 'NULL'
        publication_data.append([title, pub_year, citation, author_pub_id, num_citations, cited_by_url,
                                 cited_id, pages, publisher, cites_per_year, current_paper_authors])
        publications.append(publication_data[0])

    return publications


def get_author_params(scholar_id):
    '''

    :param scholar_id: google scholar ID of Author
    :return: list containing name, affiliation, interests, citedby, cited_by_5y, h_index, h_index_5y, i10_index,
                   i10_index_5y , cites_per_year , co_authors

    '''

    author: dict = scholarly.search_author_id(scholar_id)
    name: str = author['name']

    search_query = scholarly.search_author(name)
    author = scholarly.fill(next(search_query))  # gets more data by searching by name
    params = []

    email_domain = author['email_domain']
    affiliation: str = author['affiliation']
    interests: list = author['interests']
    citedby: int = author['citedby']
    cited_by_5y = author['citedby5y']
    h_index = author['hindex']
    h_index_5y = author['hindex5y']
    i10_index = author['i10index']
    i10_index_5y = author['i10index5y']
    cites_per_year: dict = author['cites_per_year']
    co_authors = author['coauthors']
    params.append([name, affiliation, interests, citedby, cited_by_5y, h_index, h_index_5y, i10_index,
                   i10_index_5y, cites_per_year, co_authors])
    return params[0]


def get_paper_data_from_json(json_name):
    with open(json_name, 'r') as json_file:
        data = json.load(json_file)

    publications_list = []
    for publication in data:
        publication_list = [
            publication['paper_title'],
            publication['publication_year'],
            publication['journal_info'],
            publication['author_pub_id'],
            publication['num_of_citations'],
            publication['cited_by_url'],
            publication['cites_id'],
            publication['pages'],
            publication['publisher'],
            publication['cites_per_year']
        ]
        publications_list.append(publication_list)

    return publications_list


class Author:
    def __init__(self, ID):
        try:
            print('Getting Authors data...')
            params = get_author_params(ID)
            paper_params = get_paper_params(ID)
        except AttributeError:
            raise KeyError('Invalid ID')
        self.scholar_id = ID
        self.name = params[0]
        self.affiliation = params[1]
        self.interests = params[2]
        self.citedby = params[3]
        self.citedby_5y = params[4]
        self.h_index = params[5]
        self.h_index_5y = params[6]
        self.i10_index = params[7]
        self.i10_index_5y = params[8]
        self.cites_per_year = params[9]
        self.co_authors = params[10]
        self.publications = paper_params

    def pprint_all_author_data(self, showpublications=True) -> None:

        print(f'Author Name: {self.name}')
        print(f'Autor Google Schoalr ID: {self.scholar_id}')
        print(f'Affiliation: {self.affiliation}')
        print(f'Interests: {self.interests}')
        print(f'Cited By: {self.citedby}')
        print(f'citedby_5y: {self.citedby_5y}')
        print(f'h_index: {self.h_index}')
        print(f'h_index_5y: {self.h_index_5y}')
        print(f'i10_index: {self.i10_index}')
        print(f'i10_index_5y: {self.i10_index_5y}')
        print(f'cites_per_year: {self.cites_per_year}')
        print(f'co_authors: {self.co_authors}')
        if showpublications:
            for publication in self.publications:
                print(f'publications: {publication}')
        return None

    def save_authors_paper_data_in_json(self, json_name):
        # List of keys for the JSON structure
        keys = [
            'paper_title', 'publication_year', 'journal_info', 'author_pub_id',
            'num_of_citations', 'cited_by_url', 'cites_id', 'pages',
            'publisher', 'cites_per_year', 'all_authors'
        ]

        # Convert the publications list to a list of dictionaries
        formatted_data = []
        for publication in self.publications:
            publication_dict = dict(zip(keys, publication))
            formatted_data.append(publication_dict)

        # Save the formatted data to a JSON file
        if not os.path.exists(json_name):
            with open(json_name, 'w') as file:
                json.dump(formatted_data, file, indent=4)
        else:
            with open(json_name, 'w') as file:
                json.dump(formatted_data, file, indent=4)

        print(f"Paper data has been saved to {json_name}")

    def save_authors_person_data_in_json(self, json_name):
        # Create a dictionary with the required keys
        author_data = {
            'author_name': self.name,
            'author_google_scholar_id': self.scholar_id,  # Assuming this field is not provided in the params
            'affiliation': self.affiliation,
            'interests': self.interests,
            'cited_by': self.citedby,
            'cited_by_5y': self.citedby_5y,
            'h_index': self.h_index,
            'h_index_5y': self.h_index_5y,
            'i10_index': self.i10_index,
            'i10_index_5y': self.i10_index_5y,
            'cites_per_year': self.cites_per_year,
            'co_authors': self.co_authors
        }

        # Check if the file exists; if not, create it
        if not os.path.exists(json_name):
            with open(json_name, 'w') as file:
                json.dump(author_data, file, indent=4)
        else:
            with open(json_name, 'w') as file:
                json.dump(author_data, file, indent=4)


def weight_citations_based_on_function_of_time(cites_per_year_per_paper: dict, function: str) \
        -> int:
    '''
    weighs each citation per year for a paper based on a function over time and returns the sum of all the weighted
    citations. e.g. if input funciton = e^x this function starting from publication year s.t. f(pub_year) = 1
    is multiplied by the number of citations and adds a "weight" on the citations. then the number of all
    citations is summed up.
    :param cites_per_year_per_paper: dictionary with paper as key and nested dictionary['year'] = citations as value
    e.g.  {'paper1': {'2011': 5, '2012': 3},
          'paper2': {'2010': 2, '2013': 7}...}

    :param function: function to be applied to citations of each paper. function parameter uses simple syntax e.g.
    exp((x-10)/4)
    :return: sum of weighted citations. e.g. if authors total citations are 4000, function will return 4100 after
    the function is applied to each citation and weighing it.
    '''
    weighted_citations_sum = 0
    for paper_name, citations_per_year in cites_per_year_per_paper.items():
        pub_year = int(next(iter(citations_per_year)))  # pub year of current paper
        for year, citations in citations_per_year.items():
            weighted_citations_sum += (citations * evaluate_function(function, int(year) - pub_year))
    return weighted_citations_sum


def evaluate_function(func: str, x_value: float):
    '''

    :param func: function of x in simple syntax e.g. exp((x-10)/4)
    :param x_value: x value
    :return: y value based on f(x)
    '''

    sympy_expr = sp.sympify(func, evaluate=False)

    x = sp.Symbol('x')
    # substitute x
    result = sympy_expr.subs(x, x_value)
    evaluated_result = result.evalf()

    return evaluated_result


def curve_model(x, a, b, c):
    """Model function: (ax)^b * exp(-cx)."""

    foo = a * (x ** b) * np.exp(-c * x)

    return foo


def get_gamma_distribution_best_fit_parameters(cites_per_year_per_paper) -> tuple:
    """
    
    :param cites_per_year_per_paper: nested dictionary in get_citations_per_year_per_paper function return format
    :return: a, b, c, floating parameters for a gamma distribution curve
    """
    matrix = [list(inner_dict.values()) for inner_dict in cites_per_year_per_paper.values()]
    max_ = len(matrix[0])  # finding max padding dimension

    for index, row in enumerate(matrix):
        if len(row) > max_:
            max_ = len(row)
    # padding values
    for index, value in enumerate(matrix):
        if len(value) < max_:
            for k in range(max_ - (len(value))):
                matrix[index].append(0)

    data_array = np.array(matrix).T  # we need mean for each column not row

    # Calculate mean for each row
    means = np.mean(data_array, axis=1)

    # Convert to a list (if required)
    means_list = means.tolist()
    means_list = [round(x, 4) for x in means_list]
    years = [i for i in range(1, max_ + 1)]

    # Convert input lists to numpy arrays

    x_data = np.array(years)
    y_data = np.array(means_list)

    # Initial guess for the parameters [a, b, c]
    initial_guess = [1.0, 1.0, 1.0]

    # Perform the curve fitting
    params, _ = curve_fit(curve_model, x_data, y_data, p0=initial_guess)
    a, b, c = params
    a = round(a, 4)
    b = round(b, 4)
    c = round(c, 4)

    return a, b, c


def calculate_difference_from_mean(a1, b1, c1, a2, b2, c2) -> float:
    """
    Finding the difference between 2 distributions, giving a measure of how much an author can be worse / best
    from the mean. Is calculated by finding the difference of the integrals using a Riemman Sum. The integral bounds
    are 0 and 100, we assume a paper wont get citations after 100 years for optimizations purposes.
    Note: We assume a2, b2, c2 is the mean function, s.t. a positive output means the author is better than mean,
    therefore we subtract mean (f2) from function 1.
    :param a1: Gamma distribution coefficients of function1
    :param b1:
    :param c1:
    :param a2: --- of function 2
    :param b2:
    :param c2:
    :return:
    """
    lower_integral_bound = 0
    upper_integral_bound = 100
    step_size = 0.01
    riemman_sum = 0

    for step in np.arange(lower_integral_bound, upper_integral_bound, step_size):
        y_1 = curve_model(step, a1, b1, c1)
        y_2 = curve_model(step, a2, b2, c2)
        riemman_sum += (y_1 - y_2) * step_size

    riemman_sum = round(riemman_sum, 4)
    return riemman_sum


def plot_author_citations(cites_per_year_per_paper: Dict[str, Dict[str, int]],
                          show_regression: bool = False,
                          other_authors: List[float] | List[List[float]] = None,
                          save: bool = False,
                          output_directory: str = '.',
                          file_name: str = 'author_plot.png') -> None:
    """
    Scatter plots authors citations per year. Can also fit gamma distribution curve and plot it.
    Can also plot other authors distributions curves given their function coefficients.
    Note: when taking coefficients for different authors, x-axis limit will be set to
    parameter authors oldest paper.
    :param cites_per_year_per_paper: nested dictionary with cites per year for each paper
    :param save: save figure plot
    :param show_regression: calculate and show gamma distribution curve fit on authors data
    :param other_authors: list of coefficitients for other gamma distributions
    :param output_directory: output directory of plot if save is set to True
    :param file_name: output file name.png

    :return: nothing

    """
    matrix: List[List[int]] = [list(inner_dict.values()) for inner_dict in cites_per_year_per_paper.values()]

    # plotting regular data
    for index, row in enumerate(matrix):
        x = range(len(row))
        plt.scatter(x, row, color='blue')

    # creating best curve fit and plotting it on same plot
    # ---- finding oldest paper (x-axis limit)
    step = 0.001
    max_ = len(matrix[0])
    for index, row in enumerate(matrix):
        if len(row) > max_:
            max_ = len(row)

    # x_values = [0.001, 0.002 ...
    x_values: List[float] = [round(step, 4) for step in np.arange(0, max_, step)]
    y_values: List[float] = []
    if show_regression:
        # creating data to plot
        a, b, c = get_gamma_distribution_best_fit_parameters(cites_per_year_per_paper)
        for value in x_values:
            y_values.append(round(curve_model(value, a, b, c), 3))
        # plotting on same plot
        plt.plot(x_values, y_values, color='red')

    if not (other_authors is None):
        # other authors coefficients can be a nested list or a single list
        # eg [1, 2, 3] (1 author) or [[1,2,3],[2,3,4]...] (2+ authors)
        if isinstance(other_authors[0], int | float):
            y_values = []
            for value in x_values:
                y_values.append(round(curve_model(value, other_authors[0], other_authors[1], other_authors[2]), 3))
            plt.plot(x_values, y_values, color='green')
        else:
            # nested lists case
            # using different colours for each plot
            colors = ['green', 'orange', 'red', 'blue', 'purple']
            for index, row in enumerate(other_authors):
                y_values = []
                for value in x_values:
                    y_values.append(round(curve_model(value, row[0], row[1], row[2]), 3))
                plt.plot(x_values, y_values, color=colors[index % len(colors)])
    if save:  # if output directory is not set to something it will save in working directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        filepath = os.path.join(output_directory, file_name)
        plt.savefig(filepath)

    plt.xlabel('Years after paper publication')
    plt.ylabel('Citations in year')
    plt.title('Citations per year distribution')

    plt.show()


plot_author_citations(get_citations_per_year_per_paper('papakostas_paper_data.json'), save=True)
