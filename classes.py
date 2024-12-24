from scholarly import scholarly
from analysis import get_author_params, get_paper_params
import os
import json
from colorama import Fore, Style, init
from typing import Dict, List
from scholarly import ProxyGenerator



class Author:
    def __init__(self, ID):
        try:
            init()
            print(f"{Fore.RED}Getting author {ID} data...{Style.RESET_ALL}", end='', flush=True)
            params = get_author_params(ID)
            # to clear output
            # os.system('cls' if os.name == 'nt' else 'clear')
            paper_params = get_paper_params(ID)

        except AttributeError:
            raise KeyError('Invalid ID.')

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
        # self.co_authors = params[10]
        self.publications: List = paper_params
        self.publications_instances: List[Paper] = [Paper(*paper_params[i]) for i in range(len(self.publications))]

        # most times co_authors getter doesn't work from scholarly library and returns empty list, however publications
        # data gives us the co-authors for each paper correctly , therefore we take the co_author data from the
        # publications data instead.
        co_authors = [self.publications_instances[paper].all_authors for paper in range(len(self.publications_instances))]

        # converting all paper co-authors to 1d array, and removing author's name.
        co_authors = [item for sublist in co_authors for item in sublist if item != self.name]

        def normalize_name(name):
            # self.name and publication name might be slightly different, only case i found is dot case
            # e.g. Kostas K. Vandlopoulos == Kostas K Vandlopoulos, therefore we normalize by removing dots.
            return name.replace('.', '').strip()

        co_authors = [name for name in co_authors if normalize_name(name) != normalize_name(self.name)]

        # converting to set in case of many publications with same authors
        self.co_authors: set[str] = set(co_authors)

        co_author_ids  = []
        # getting co-authors ids
        for co_author in self.co_authors:
            search_query = scholarly.search_author(co_author)
            # Retrieve the first result from the iterator
            try:
                first_author_result = next(search_query)
            except StopIteration:
                break
            # Retrieve all the details for the author
            author = scholarly.fill(first_author_result)
            co_author_ids.append(author['scholar_id'])
        self.co_author_ids: List[str] = co_author_ids

    def pprint_all_author_data(self, show_publications=True) -> None:

        print(f'Author Name: {self.name}')
        print(f'Autor Google Scholar ID: {self.scholar_id}')
        print(f'Affiliation: {self.affiliation}')
        print(f'Interests: {self.interests}')
        print(f'Cited By: {self.citedby}')
        print(f'citedby_5y: {self.citedby_5y}')
        print(f'h_index: {self.h_index}')
        print(f'h_index_5y: {self.h_index_5y}')
        print(f'i10_index: {self.i10_index}')
        print(f'i10_index_5y: {self.i10_index_5y}')
        print(f'Cites per year: {self.cites_per_year}')
        print(f'Co-Authors list: {self.co_authors}')
        if show_publications:
            for index, publication in enumerate(self.publications):
                print(f'Publication {index + 1}: {publication}')
        return None

    def save_authors_paper_data_in_json(self, json_name=None):
        if json_name is None:
            json_name = self.name + '.json'
        elif '.json' not in json_name:
            raise ValueError('Path should end in .json.')

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

        if '.json' not in json_name:
            raise ValueError('Path should end in .json.')

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


class Paper:
    def __init__(self,
                 title: str,
                 pub_year: str,
                 journal_info: str,
                 author_pub_id: str,
                 citations: str,
                 cited_by_url: str,
                 cites_id: str,
                 pages: str,
                 publisher: str,
                 cites_per_year: Dict[str, int],
                 all_authors: List[str]):

        self.title = title
        self.paper_title = title
        self.publication_year = pub_year
        self.journal_info = journal_info
        self.author_pub_id = author_pub_id
        self.num_of_citations = citations
        self.cited_by_url = cited_by_url
        self.cites_id = cites_id
        self.pages = pages
        self.publisher = publisher
        self.cites_per_year = cites_per_year
        self.all_authors = all_authors

    def get_paper_data(self):
        return (f"ResearchPaper("
                f"paper_title={self.paper_title}, "
                f"publication_year={self.publication_year}, "
                f"journal_info={self.journal_info}, "
                f"author_pub_id={self.author_pub_id}, "
                f"num_of_citations={self.num_of_citations}, "
                f"cited_by_url={self.cited_by_url}, "
                f"cites_id={self.cites_id}, "
                f"pages={self.pages}, "
                f"publisher={self.publisher}, "
                f"cites_per_year={self.cites_per_year}, "
                f"all_authors={self.all_authors})")
