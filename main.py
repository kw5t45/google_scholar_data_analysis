from scholarly import scholarly
import json
from tqdm import tqdm
import sys
import os
import sympy as sp
import math

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

    def weight_citations_based_on_function_of_time(self, cites_per_year_per_paper: dict, function: str)\
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
        obj = Analysis_Tools()
        weighted_citations_sum = 0
        for paper_name, citations_per_year in cites_per_year_per_paper.items():
            pub_year = int(next(iter(citations_per_year))) # pub year of current paper
            for year, citations in citations_per_year.items():
                weighted_citations_sum += (citations * obj.evaluate_function(function, int(year) - pub_year))
        return weighted_citations_sum

    def evaluate_function(self, func: str, x_value: float):
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


# id enos tyxaioy me liga papers jP1qgO4AAAAJ
# id Papakwsta O9d4j7oAAAAJ
#tyxaios = Author('O9d4j7oAAAAJ')
#tyxaios.save_authors_paper_data_in_json('papakostas_paper_data.json')
#o = Analysis_Tools()
#print(o.evaluate_function(r'exp((x-10)/4)', 1))
analysis_obj = Analysis_Tools()
tools_obj = Tools()
e = math.e
sum = analysis_obj.weight_citations_based_on_function_of_time(tools_obj.get_citations_per_year_per_paper('papakostas_paper_data.json'), fr'{e}^(-(x)^2)')
print(sum)