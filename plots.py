from analysis import get_gamma_distribution_best_fit_parameters, curve_model
import json
from typing import List, Dict, Tuple
import os
from exceptions import check_dictionary_format
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from analysis import _clean_author_name_for_plotting


def plot_author_citations(cites_per_year_per_paper: Dict[str, Dict[str | int, int]],
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
    assert check_dictionary_format(cites_per_year_per_paper), r"Invalid dictionary format. Dictionary should match " \
                                                              r"get_citations_per_year_per_paper function's output " \
                                                              r"format."

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


def get_co_author_graph_pairs(path: str) -> List[List[str]]:
    """
    Takes in a folder created from checkpoint_save_author_and_coauthors_in_tree() function and returns
    co-author list of pairs
    this function only returns and doesnt plot, so that multiple lists can be merged toghether to create a bigger
    more general graph.
    This function works for depth 2 tree only, meaning the folder is in
    - Author
    - author_data
    - Co_author
    --- coauthor data
    ------- co author's co authors
    example tree / folder:
    depth = 0                            Author (folder)
                                        /    \
                                      /       \
    depth = 1                    Co-author   Co-author (folder)
                                /         \   /       \
    depth = 2              Co-author      Co-author  Co-author (list of strings in above folder)

    Since in last depth (=2) the coauthors are known as strings, and finding their IDs will take an enormous amount of
    time, here we plot the names instead.
    :param path: full path to folder containing co author tree
    :return: list of pairs
    """
    ####
    # check integrity of folder here etc
    ####
    all_authors_in_tree: List[str] = []

    # depth = 1
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # Check if it's a JSON file
        if item.endswith('author_data.json'):
            with open(item_path, 'r') as f:
                data = json.load(f)
    all_authors_in_tree.append(data['author_name'])

    # depth = 2
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)

                # Check if it's a JSON file
                if item.endswith('author_data.json'):
                    with open(item_path, 'r') as f:
                        data = json.load(f)
                        all_authors_in_tree.append(data['author_name'])
                        all_authors_in_tree.extend(data['co_authors'])

    all_pairs: List[List[str]] = []

    for folder in os.listdir(path):
        if folder.endswith("author_data.json"):  # getting root author name
            with open(os.path.join(path, folder), 'r') as f:
                root_name = json.load(f)['author_name']

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                # Check if it's a JSON file
                if item.endswith('author_data.json'):
                    with open(item_path, 'r') as f:
                        data = json.load(f)
                        coa_name: str = data['author_name']  # co-author name, starting index
                        # creating depth 0 (root) - depth 1 pairs
                        all_pairs.append([root_name, coa_name])
                        # creating depth 1 - depth 2 pairs
                        # here root becomes coa_name variable, we do a similar process
                        for d2_coauthor in data['co_authors']:
                            all_pairs.append([coa_name, d2_coauthor])

    for i in range(len(all_pairs)):
        for j in range(len(all_pairs[0])):
            all_pairs[i][j] = _clean_author_name_for_plotting(all_pairs[i][j], last_name_only=True)
    print(len(all_pairs))
    return all_pairs


def plot_co_author_graph(co_author_pairs: List[List[str]], **kwargs) -> None:
    """
    :param kwargs: key word parameters to be passed into networkx.draw() function
    :param co_author_pairs: List of pairs as retrieved from get_co_author_graph_pairs() function
    :return: Plots graph of co-author relationships

    """
    # Create a graph object
    G = nx.Graph()

    # Add edges to the graph
    G.add_edges_from(co_author_pairs)
    # , node_size=350, node_color="lightblue", font_size=10, font_weight="bold",
    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True,node_size=100, node_color="lightblue", font_size=5, width=0.1, alpha=0.8, **kwargs)
    plt.title("Graph Representation")
    plt.show()

