from scholarly import scholarly
import json
from tqdm import tqdm
import sys
import sympy as sp
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple
import os
from exceptions import raise_max_requests_exceeded, check_dictionary_format
import networkx as nx
import matplotlib.pyplot as plt


def get_citations_per_year_per_paper(json_name: str) -> dict:
    '''

    :param json_name: json file containing authors paper data
    :return: dictionary in form:
         {'paper1': {'2011': 5, '2012': 3},
          'paper2': {'2010': 2, '2013': 7}...}
          where nested dictionary is the citations per year
    '''
    if '.json' not in json_name:
        raise ValueError('Path should end in .json.')

    if not os.path.isfile(json_name):
        raise FileNotFoundError(f"The file '{json_name}' does not exist.")

    paper_data: list = get_paper_data_from_json(json_name)  # list of all papers
    dic = {}
    for paper in paper_data:
        dic[paper[0]] = paper[-1]

    return dic


def get_paper_params(scholar_id, clip: int=-1):
    '''
    Given a Unique google scholar profile ID this function fetches paper data which are described below.
    :param scholar_id:
    :param clip: number of publications to be returned. This is because after ~700 publications with a fresh ip we get
    requests denied error.
    :return: (papers)  [title, pub_year, citation, author_pub_id, num_citations, cited_by_url,
                                 cited_id, pages, publisher, cites_id, cites_per_year]
    '''
    try:
        author: dict = scholarly.search_author_id(scholar_id)
    except:
        raise_max_requests_exceeded()
    name: str = author['name']
    try:
        search_query = scholarly.search_author(name)
        author = scholarly.fill(next(search_query))  # gets more data by searching by name
    except:
        raise_max_requests_exceeded()
    # cleaning Author's publication data

    publications = []
    # keeping only first <clip> publications
    if clip != -1:
        author['publications'] = author['publications'][:clip]

    for index, value in tqdm(enumerate(author['publications']), desc=f'Getting author {scholar_id} paper data',
                             total=len(author['publications'])):
        try:
            current_iterable_publication = scholarly.fill(author['publications'][index])
        except:
            raise_max_requests_exceeded()
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
        try:
            current_paper_authors: str = current_iterable_publication['bib']['author']  # NOQA
            current_paper_authors: list = current_paper_authors.split(" and ")
        except KeyError:
            current_paper_authors: List = []
        try:
            pub_year = current_iterable_publication['bib']['pub_year']
        except KeyError:
            pub_year = 'NULL'
        try:
            citation = current_iterable_publication['bib']['citation']
        except KeyError:
            citation = 'NULL'
        try:
            author_pub_id = current_iterable_publication['author_pub_id']
        except KeyError:
            author_pub_id = 'NULL'
        try:
            num_citations = current_iterable_publication['num_citations']
        except KeyError:
            num_citations = 'NULL'
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

        if pub_year != 'NULL':
            # padding 0 values in years that got no citations, from publication year up to current year
            for year in range(pub_year, 2024 + 1):
                if (year not in cites_per_year) or (str(year) not in cites_per_year):
                    cites_per_year[str(year)] = 0

            # converting all keys to strings and sorting
            cites_per_year = {str(key): value for key, value in cites_per_year.items()}
            cites_per_year = dict(sorted(cites_per_year.items(), key=lambda item: item[0]))
        else:
            cites_per_year = {}
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
    try:
        search_query = scholarly.search_author(name)
        author = scholarly.fill(next(search_query))  # gets more data by searching by name
    except:
        raise_max_requests_exceeded()
    params = []
    # here we try except everything because google returns random stuff, doesnt return none so we must hard-code them
    # in case they dont appear
    try:
        email_domain: str = author['email_domain']
    except KeyError:
        email_domain: str = 'No Email found'
    try:
        affiliation: str = author['affiliation']
    except KeyError:
        affiliation: str = 'No Affiliation found'

    try:
        interests: list = author['interests']
    except KeyError:
        interests: list = []

    try:
        citedby: int = author['citedby']
    except KeyError:
        citedby: int = 0

    try:
        cited_by_5y: int = author['citedby5y']
    except KeyError:
        cited_by_5y: int = 0

    try:
        h_index: int = author['hindex']
    except KeyError:
        h_index: int = 0

    try:
        h_index_5y: int = author['hindex5y']
    except KeyError:
        h_index_5y: int = 0

    try:
        i10_index: int = author['i10index']
    except KeyError:
        i10_index: int = 0

    try:
        i10_index_5y: int = author['i10index5y']
    except KeyError:
        i10_index_5y: int = 0

    try:
        cites_per_year: dict = author['cites_per_year']
    except KeyError:
        cites_per_year: dict = {}

    try:
        co_authors: list = author['coauthors']
    except KeyError:
        co_authors: list = []
    params.append([name, affiliation, interests, citedby, cited_by_5y, h_index, h_index_5y, i10_index,
                   i10_index_5y, cites_per_year, co_authors])
    return params[0]


def get_paper_data_from_json(json_name) -> List[List]:
    """

    :param json_name:
    :return: returns list of publications
    """

    if '.json' not in json_name:
        raise ValueError('Path should end in .json.')

    if not os.path.isfile(json_name):
        raise FileNotFoundError(f"The file '{json_name}' does not exist.")

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


def weight_citations_based_on_function_of_time(cites_per_year_per_paper: dict, function: str) \
        -> float:
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
    assert check_dictionary_format(cites_per_year_per_paper), r"Invalid dictionary format. Dictionary should match " \
                                                              r"get_citations_per_year_per_paper function's output " \
                                                              r"format."

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


def curve_model(x: float | int, a: float | int, b: float | int, c: float | int) -> float | int:
    """Model function: (ax)^b * exp(-cx). (gamma distribution function)"""

    foo = a * (x ** b) * np.exp(-c * x)

    return foo


def get_gamma_distribution_best_fit_parameters(cites_per_year_per_paper) -> tuple:
    """

    :param cites_per_year_per_paper: nested dictionary in get_citations_per_year_per_paper function return format
    :return: a, b, c, floating parameters for a gamma distribution curve
    """
    assert check_dictionary_format(cites_per_year_per_paper), r"Invalid dictionary format. Dictionary should match " \
                                                              r"get_citations_per_year_per_paper function's output " \
                                                              r"format."

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


def calculate_difference_from_mean(a1: float | int,
                                   b1: float | int,
                                   c1: float | int,
                                   a2: float | int,
                                   b2: float | int,
                                   c2: float | int) -> float | int:
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


def checkpoint_save_author_and_coauthors_in_tree(identifier: str, full_path: str = 'bar', clip: int=-1) -> None:
    """
    Creates nested file in form:
    folder: parameter_author which contains a json with author data (and publication data) and a nested folder
    co-authors: a folder containing json files with co-authors data of parameter author.
    Because a lot of the requests get timed out and i cant figoure out proxies as of current version, this function saves
    fetched data and if it crashes, it can be called again and continue the process on the half saved folder. this
    theoretically could be continued for level 2 and so on trees, however it would take exponentially more time and
    we would get requests denied. As of current version depth is equal to 1.
    Example output folder format:
    - ID_author
    --- author_name_author_data.json
    --- author_name_paper_data.json
    --- ID_co-author
    ----- co_author_name_author_data.json
    ----- co_author_name_paper_data.json
    --- ID_2nd_co-author
    Note:
    - in author data for each entity (co-author) we do not include their co-authors as IDs but as names only.
    This is beacuse the names of co-authors are taken from publications, but to get IDs the names must be searched
    one by one, and this takes a LOT of time. Thus in graph analysis we use names, not IDs.
    - make sure input path folder is empty, or has data that were saved if the code had stopped. If folder was not empty
    the code assumes that it must "continue" the data fetching process and will break.
    :param identifier: unique author ID
    :param full_path: full path of folder, working directory by default
    :param clip: number of publications to be returned. This is beacuse after ~700 publications with a fresh IP the code
    still crashes.
    :return:

    """
    # to avoid circular import crash
    import classes as gs

    # file already exists boolean, we assume file does not exist. If it does (code crashed somepoint) the code works
    # normally.
    exists = False

    output_folder: str = f'{identifier}'

    if full_path != 'bar':
        output_folder = os.path.join(output_folder, full_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Parent folder {output_folder} created successfully.")
    else:
        print(f"Folder {output_folder} already exists, proceeding to continue fetching data.")
        exists = True
        if not os.listdir(output_folder):  # Empty folder has no items - process stopped in parent folder data fetching
            print(f"The folder '{output_folder}' is empty (Process might have stopped in author data fetching).\n"
                  f"(re)starting parent folder process: ")
            exists = False  # restarting process
    if not exists:

        author: gs.Author = gs.Author(identifier, clip=clip)

        # HERE MAKE IT FULL PATH, NOT STARTING WITH ID here parent folder will have ID as name, since it is created
        # before fetching data (name). I could rename it later.
        author.save_authors_paper_data_in_json(
            f'{output_folder}/{author.name.replace(" ", "_").lower()}_paper_data.json')  # NOQA, if code reaches here author var exists
        author.save_authors_person_data_in_json(
            f'{output_folder}/{author.name.replace(" ", "_").lower()}_author_data.json')

        # creating a .json file with co-author IDs. This will be later helpful for continuing the search if the code
        # breaks when searching a co-author.
        co_authors_list_file = os.path.join(output_folder, "co_authors.json")
        with open(co_authors_list_file, 'w') as f:
            json.dump(author.co_author_ids, f)

        # here we create a clone file, so that we can pop IDs and also keep track of progress (if code stops)
        co_authors_list_file__ = os.path.join(output_folder, "co_authors_to_search.json")
        with open(co_authors_list_file__, 'w') as f:
            json.dump(author.co_author_ids, f)
    # here the code is most likely to fail and exceed max tries. ##

    # in this point the ID author folder is
    # - ID_author
    # --- author_name_author_data.json
    # --- author_name_paper_data.json
    # we make the folder a tree in this form (for comment simplicity, here author has only 2 co-authors):
    # > ID_author
    # >>> co_authors.json
    # >>> co_authors_to_remove.json
    # >>> author_name_author_data.json
    # >>> author_name_paper_data.json
    # >>> ID_co-author
    # >>>>> co_author_name_author_data.json
    # >>>>> co_author_name_paper_data.json
    # >>> ID_2nd_co-author
    # ...
    # this way we can create a simple nested folder which can work as a tree, and we can retrieve data and perform
    # analysis later.
    # this function could be called recursively, however in current version i dont do that as i dont think it is needed.
    # recursion would be ideal when building trees of bigger depth, however this is nearly impossible with current
    # data fetching library.

    # get length of all authors to be searched, if code is re-excecuted some of them have already been searched.
    with open(f'{os.path.join(output_folder)}\\co_authors.json', 'r') as f:
        ids: list[str] = json.load(f)
    co_authors_number: int = len(ids)

    with open(f'{output_folder}\\co_authors_to_search.json', 'r') as f:
        ids: list[str] = json.load(f)
    co_authors_current: int = len(ids)

    progress = 0
    if exists:
        progress = round(100 * (co_authors_number - co_authors_current) / co_authors_number)
        print(f'Total progress: {progress}%. ({co_authors_number-co_authors_current}/{co_authors_number})')

    # here we use a copy of IDs list because we remove each iterated ID from the list.
    for co_author_id in ids[:]:

        # creating nested folder in parent folder
        try:
            os.makedirs(f'{os.path.join(output_folder, co_author_id)}')
        except FileExistsError:
            pass
        try:
            co_author: gs.Author = gs.Author(co_author_id, get_co_authors=False, clip=clip)
        except:
            raise_max_requests_exceeded()
        co_author.save_authors_paper_data_in_json(  # NOQA, if code reaches here co_author var exists
            f'{output_folder}\\{co_author_id}\\{co_author.name.replace(" ", "_").lower()}_paper_data.json')
        co_author.save_authors_person_data_in_json(
            f'{output_folder}\\{co_author_id}\\{co_author.name.replace(" ", "_").lower()}_author_data.json')

        ids.remove(co_author_id)
        co_authors_current = len(ids)
        with open(f'{output_folder}\\co_authors_to_search.json', 'w') as f:
            json.dump(ids, f)
        progress = round(100 * (co_authors_number - co_authors_current) / co_authors_number)
        print(f'Total progress: {progress}%. ({co_authors_number-co_authors_current}/{co_authors_number})')

    # when all data is fetched we remove useless json files from folder
    if int(progress) == 100:
        co_authors_json = os.path.join(output_folder, "co_authors.json")
        if os.path.exists(co_authors_json):
            os.remove(co_authors_json)
        co_authors_to_search_json = os.path.join(output_folder, "co_authors_to_search.json")
        if os.path.exists(co_authors_to_search_json):
            os.remove(co_authors_to_search_json)

        print(f'{identifier} co-author tree saved successfully.')


def _clean_author_name_for_plotting(name: str, last_name_only=True) -> str:
    """
    Takes an input string such as
    Vandlopoulos K. Ioannis
    and returns
    VK Ioannis.
    This is because in many cases there are duplicates of a name for no reason and in graph plotting and analysis
    they are taken for 2 different authors.
    :param last_name_only: return only last name
    :param name: author name as fetched from scholar
    :return: cleaned string
    """
    parts = name.split()
    if last_name_only:
        return parts[-1]
    if len(parts) == 1:
        return parts[0]
    return "".join([p[0] for p in parts[:-1]]) + " " + parts[-1]

