from scholarly import scholarly
import json
from tqdm import tqdm
import sys
import os
import sympy as sp
import math
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import numpy as np


class Tools:

    def __innit__(self):
        pass

    def get_citations_per_year_per_paper(self, json_name) -> dict:
        '''

        :param json_name: json file containing authors paper data
        :return: dictionary in form:
             {'paper1': {'2011': 5, '2012': 3},
              'paper2': {'2010': 2, '2013': 7}...}
              where nested dictionary is the citations per year
        '''
        tool = Tools()
        paper_data: list = tool.get_paper_data_from_json(json_name)  # list of all papers
        dic = {}
        for paper in paper_data:
            dic[paper[0]] = paper[-1]

        return dic

    def get_paper_params(self, scholar_id):
        '''

        :param scholar_id:
        :return: (papers)  [title, pub_year, citation, author_pub_id, num_citations, cited_by_url,
                                     cited_id, pages, publisher, cites_id, cites_per_year]
        '''
        author: dict = scholarly.search_author_id(scholar_id)
        name: str = author['name']
        search_query = scholarly.search_author(name)
        author = scholarly.fill(next(search_query))  # gets more data by searching by name
        progress_bar = tqdm(total=len(author['publications']), desc="Processing", unit="iteration", leave=False)

        # cleaning Author's publication data

        publications = []

        for index, value in enumerate(author['publications']):
            # index += 251 # used for getting data in batches after changing VPN to avoid too many requests errors
            # if index == 500:
            #     break
            current_iterable_publication = scholarly.fill(author['publications'][index])
            # progress bar updating
            progress_bar.update(1)
            if index % 10 == 0:
                # Clear the line for visual effect
                sys.stdout.write("\033[K")
                # Re-display the progress bar
                progress_bar.display()
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

            try:
                pub_year = current_iterable_publication['bib']['pub_year']
            except:
                pub_year = 2010  # all values will be padded as 0 anyway
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
                cites_per_year = {}

            # padding 0 values in years that got no citations, from publication year up to current year
            # Generate a list of years to add
            years_to_add = [year for year in range(pub_year, 2024 + 1) if year not in cites_per_year]
            # Merge the existing keys and the years to add, then sort them
            keys = [key for key in cites_per_year.keys()]
            for i in years_to_add:
                keys.append(i)
            sorted_keys = sorted(keys)
            # Create the new dictionary with zero values for the missing years
            new_d = {key: cites_per_year.get(key, 0) for key in sorted_keys}

            cites_per_year = new_d

            try:
                cited_id = current_iterable_publication['cites_id'][0]  # is list for some reason

            except KeyError:  # some publications have no cites id or url for some reason
                cited_by_url = 'NULL'
                cited_id = 'NULL'
            publication_data.append([title, pub_year, citation, author_pub_id, num_citations, cited_by_url,
                                     cited_id, pages, publisher, cites_per_year, current_paper_authors])
            publications.append(publication_data[0])

        progress_bar.close()

        return publications

    def get_author_params(self, scholar_id):
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

    def get_paper_data_from_json(self, json_name):
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
        obj = Tools()

        params = obj.get_author_params(ID)
        paper_params = obj.get_paper_params(ID)
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

    def pprint_all_author_data(self, showpublications=True):
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
            'author_google_scholar_id': '',  # Assuming this field is not provided in the params
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


class Analysis_Tools:
    def __init__(self):
        pass

    def weight_citations_based_on_function_of_time(self, cites_per_year_per_paper: dict, function: str) \
            -> tuple:
        """
        weighs each citation per year for a paper based on a function over time and returns the sum of all the weighted
        citations. e.g. if input funciton = e^x this function starting from publication year s.t. f(pub_year) = 1
        is multiplied by the number of citations and adds a "weight" on the citations. then the number of all
        citations is summed up.
        :param cites_per_year_per_paper: dictionary with paper as key and nested dictionary['year'] = citations as value
        e.g.  {'paper1': {'2011': 5, '2012': 3},
              'paper2': {'2010': 2, '2013': 7}...}

        :param function: function to be applied to citations of each paper. function parameter uses simple syntax e.g.
        exp((x-10)/4)
        :return: tuple contatining normal citations and
        sum of weighted citations. e.g. if authors total citations are 4000, function will return (4000, 4100) after
        the function is applied to each citation and weighing it.
        """
        obj = Analysis_Tools()
        weighted_citations_sum = 0
        total_citations = 0
        for paper_name, citations_per_year in cites_per_year_per_paper.items():
            pub_year = int(next(iter(citations_per_year)))  # pub year of current paper
            for year, citations in citations_per_year.items():
                weighted_citations_sum += (citations * obj.evaluate_function(function, int(year) - pub_year))
                total_citations += citations
        return total_citations, weighted_citations_sum

    def evaluate_function(self, func: str, x_value: float):
        """

        :param func: function of x in simple syntax e.g. exp((x-10)/4)
        :param x_value: x value
        :return: y value based on f(x)
        """

        sympy_expr = sp.sympify(func, evaluate=False)

        x = sp.Symbol('x')
        # substitute x
        result = sympy_expr.subs(x, x_value)
        evaluated_result = result.evalf()

        return evaluated_result

    def plot_author_citations(self, citations_per_year_per_paper, author=' ') -> None:
        """
        Function plots number of citations on Y axis and years after paper publication on X axis.
        :param citations_per_year_per_paper: dicitonary of citations per year per paper
        :return: None
        """

        x_y_pairs = []
        # converting data to (x, y) pairs
        for paper_name, citations_per_year in citations_per_year_per_paper.items():
            pub_year = int(next(iter(citations_per_year)))  # pub year of current paper
            for year, citations in citations_per_year.items():
                pair = ((int(year) - pub_year + 1), citations)
                x_y_pairs.append(pair)

        x_values = [pair[0] for pair in x_y_pairs]
        y_values = [pair[1] for pair in x_y_pairs]

        # Plotting the data as points
        plt.scatter(x_values, y_values, label='', s=10)  # Adjust the size as needed

        # Adding labels
        plt.xlabel('Years after paper publication')
        plt.ylabel('Number of Citations')
        plt.title(f'Life of paper vs Citations {author}')
        # Setting x-axis limit to start from 0
        plt.xlim(0)
        plt.ylim(0)
        plt.grid(True)  # Optional: add grid for better visualization
        plt.show()

    def find_line_of_best_fit(self, citations_per_year_per_paper, plot=True) -> tuple[float]:
        """
        Given an f(x) function with any parameters, we find the optimized parameters that fit best the given dataset.
        plotting the function plots both the dataset and the function curve at the same graph.
        ##### FUNCTION IS HARD CODED AS  a * x ** b * np.exp(-c * x) inside function definition

        :param citations_per_year_per_paper: dictionary containing citations per year for each paper
        :param plot: plot the found function or not
        :return: list of optimized (best fit) parameters
        """

        x_y_pairs = []
        # converting data to (x, y) pairs
        for paper_name, citations_per_year in citations_per_year_per_paper.items():
            pub_year = int(next(iter(citations_per_year)))  # pub year of current paper
            for year, citations in citations_per_year.items():
                pair = ((int(year) - pub_year + 1), citations)
                x_y_pairs.append(pair)

        # creating x, y pairs
        x = [pair[0] for pair in x_y_pairs]
        y = [pair[1] for pair in x_y_pairs]

        for i in x:  # checking data validity
            if i < 0:
                raise ValueError('Some of the data is wrong. Publication year in some papers is bigger than the'
                                 'first recorded citation year.')

        obj = Analysis_Tools()

        # Use curve_fit from scipy to fit the custom function
        popt, _ = curve_fit(obj.custom_function, x, y)

        if plot:
            # Plotting
            # Plot the dataset
            plt.scatter(x, y, label='Dataset')

            # Generate x values for the function plot
            x_values = np.linspace(min(x), max(x), 100)

            # Calculate y values using the custom function
            y_values = obj.custom_function(x_values, popt[0], popt[1], popt[2])

            # Plot the function
            plt.scatter(x_values, y_values, label='', s=10)

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Dataset and Best fit curve: {round(popt[0], 3)}x^({round(popt[1], 3)}) e^({round(popt[2], 3)}x)'
                      f'')
            plt.show()

        # return optimal params
        popt = tuple(popt)
        return popt

    def custom_function(self, x: float, a: float, b: float, c: float) -> float:
        max_exp_argument = 700  # This is a large number to prevent overflow in np.exp
        exp_argument = -c * x
        exp_argument = np.clip(exp_argument, -max_exp_argument, max_exp_argument)
        return (a * x ** b) * np.exp(exp_argument)

    def calculate_difference_from_mean(self,
                                       mean_a: float,
                                       mean_b: float,
                                       mean_c: float,
                                       o_a: float,
                                       o_b: float,
                                       o_c: float) -> float:
        """
        all parameters describe a * x ** b * np.exp(-c * x) function of x.
        :param mean_a: a of mean function
        :param mean_b: b of --
        :param mean_c: c of --
        :param o_a: a of optimized parameter found through line_of_best_fit function
        :param o_b: b --
        :param o_c: c --
        :return: difference of mean function integral from given function integral..
        """
        obj = Analysis_Tools()
        riemman_sum = 0
        lower_bound = 0
        upper_bound = 1000  # should tend to infinity, set to 1000 for reducing time complexity -also it is unlikely
        # a paper will get citations more than 1000 years later...
        x = lower_bound
        step = 0.01  # step should tend to 0
        while x <= upper_bound:
            riemman_sum += (obj.custom_function(x, mean_a, mean_b, mean_c) - obj.custom_function(x, o_a, o_b,
                                                                                                  o_c)) * step
            x += step
        return riemman_sum



