
# Google Scholar Author Analysis

## 
This library scrapes Google Scholar profiles to fetch author data, including publications, citations, and co-authors, saving it in JSON format. It offers tools for weighted citation analysis, regression modeling of citation trends, and deviation analysis. Users can visualize citation data, highlighting research impact and temporal trends in scholarly citations.

# Documentation
### Getting started - Author Class
Create an Author instance 
```py
import <to_name> as gs

author = gs.Author('id') # where id is the unique id found in Google Scholar profile page
```
When creating an author, author data is fetched and saved in these parameters:
```py
self.scholar_id: str # Unique Author ID
self.name: str # Author full name
self.affiliation: # Author description, eg. Machine Learning engineer
self.interests: str # Author interests from google scholar bio
self.citedby: int # Cited by
self.citedby_5y: int # Cited by in the last 5 years
self.h_index: int # h-index, meaning (eg. h=10 implied 10 publications with at least 10 citations each)
self.h_index_5y: int # 5 year h-index
self.i10_index: int # do not remember what that does will fix
self.i10_index_5y: int # --
self.cites_per_year: dict[int: int] # Cites per year dictionary eg {2010: 1, 2011: 0...} 
self.co_authors: list[str] # Co-Authors
self.publications: list[list[]] # List of publications, and data for each publication, such as _______
```

### Author Class methods
##### pprint_all_author_data(showpublications=True):
Pretty prints all fetched autor's data, showing publications based on parameter (set to True by default).
Example usage:
```py
Author.pprint_all_author_data(showpublications=False)
>>>
Author Name: Kostas Vandlopoulos
Autor Google Schoalr ID: XXXXXXXXXXXX
Affiliation: Data Analyst
Interests: ['Artificial Intelligence', 'Machine Learning', 'Brain Computer Interfaces']
Cited By: 81
citedby_5y: 81
h_index: 3
h_index_5y: 3
i10_index: 2
i10_index_5y: 2
cites_per_year: {2021: 4, 2022: 17, 2023: 27, 2024: 31}
co_authors: [] # Sometimes doesnt really work for some reason will fix
publications: ['Predictive maintenance-bridging artificial intelligence and IoT', 2021, '2021 IEEE World AI IoT Congress (AIIoT), 0413-0419, 2021', 'jP1qgO4AAAAJ:u5HHmVD_uO8C', 44, '/scholar?hl=en&cites=1217316688053608473', '1217316688053608473', '0413-0419', 'IEEE', {2021: 3, 2022: 8, 2023: 14, 2024: 18}, ['Gerasimos G Samatas', 'Seraphim S Moumgiakmas', 'George A Papakostas']]

```
##### save_authors_person_data_in_json(output_path)
Saves Author's personal data in json file:
```py
Author.save_authors_person_data_in_json('output.json')
>>>
# Example format
{
    "author_name": "Vandl",
    "author_google_scholar_id": "XXXXXXXXXXXX",
    "affiliation": "Data Analyst, MLV Research Group",
    "interests": [
        "Artificial Intelligence",
        "Machine Learning",
        "Motor Imagery",
        "Brain Computer Interfaces",
        "EEG"
    ],
    "cited_by": 81,
    "cited_by_5y": 81,
    "h_index": 3,
    "h_index_5y": 3,
    "i10_index": 2,
    "i10_index_5y": 2,
    "cites_per_year": {
        "2021": 4,
        "2022": 17,
        "2023": 27,
        "2024": 31
    },
    "co_authors": []
}
```
##### save_authors_paper_data_in_json(output_path)
Saves author publication data in json file:
```
Author.save_authors_paper_data_in_json('output.json')
>>>
# Example format
[
    {
        "paper_title": "Predictive maintenance-bridging artificial intelligence and IoT",
        "publication_year": 2021,
        "journal_info": "2021 IEEE World AI IoT Congress (AIIoT), 0413-0419, 2021",
        "author_pub_id": "XXXXXXX:XXXXX",
        "num_of_citations": 44,
        "cited_by_url": "/scholar?hl=en&cites=XXXXX",
        "cites_id": "XXXXXXX",
        "pages": "0413-0419",
        "publisher": "IEEE",
        "cites_per_year": {
            "2021": 3,
            "2022": 8,
            "2023": 14,
            "2024": 18
        },
        "all_authors": [
            "Tim",
            "Sim",
            "Bartholomew"
        ]
    },
]
```
#### JSON data fetching functions
##### get_paper_data_from_json(path)
Given a json file created with *save_authors_paper_data_in_json* method, returns a nested list of basic data for each publication.
**Note these are functions and not a static methods on Author Class**
Example usage and output:
```py
data: [list[list[]] = get_paper_data_from_json('test.json')
print(data)
>>>
[['Predictive maintenance-bridging artificial intelligence and IoT', 2021, '2021 IEEE World AI IoT Congress (AIIoT), 0413-0419, 2021', 'jP1qgO4AAAAJ:u5HHmVD_uO8C', 44, '/scholar?hl=en&cites=1217316688053608473', '1217316688053608473', '0413-0419', 'IEEE', {'2021': 3, '2022': 8, '2023': 14, '2024': 18}], ...]
```

##### get_paper_params(id)
Returns paper data for author, **retrieving-fetcing from ID** and not opening from json file.
Example usage and output:
```py
data: [list[list[]] = get_paper_params('xxxxxxxx')
print(data)
>>>
[#same as above window]
```
##### get_author_params(id)
Returns Author's personal data **list** given an ID, without having to create Author instance.
Example usage and output:
```py
params: [list[]] = get_author_params('xxxxxxxxxxx')
print(params)
>>>
['Vandlopoulos Kostas', # Name
'Data Analyst, # Affiliation 
MLV Research Group', 
['Artificial Intelligence','Machine Learning'], # Interests
81, # Cited by
81, # Cited by 5-year
3, # h_index
3, # h_index_5y
2, i10
2, i10_5year
{2021: 4, 2022: 17, 2023: 27, 2024: 31}, # Citations per year
[]] # Co-authors
```
##### get_citations_per_year_per_paper(json_path)
Returns dictionary of Paper name as key, and dictionary of citations per year as value. Is callable only on json as of current version.
```py
foo: dict[str: dict[str: int]] = get_citations_per_year_per_paper('path.json')
print(foo)
>>>
{'Predictive maintenance-bridging artificial intelligence and IoT': {'2021': 3, '2022': 8, '2023': 14, '2024': 18}, 'Computer vision for fire detection on UAVs—From software to hardware': {'2021': 1, '2022': 9, '2023': 9, '2024': 12}, 'Robustly effective approaches on motor imagery-based brain computer interfaces': {'2023': 4, '2024': 1, '2022': 0}, 'Benchmarking convolutional neural networks on continuous EEG signals: The case of motor imagery–based BCI': {}}
```
### Analysis functions
##### weight_citations_based_on_function_of_time(cites_per_year_per_paper: dict[str: dict[str: int]], function: str)

Using **weight_citations_based_on_function_of_time**, we can "weigh" all of the citations of an author 
based on how **recent** the citations are compared to how **old** the paper is, using:

$$\sum_{p=1}^{papers}\sum_{n=y_0}^{y_c}f(n-y_n)c(p, n)$$

Where:
- **papers** is the number of all the author's publications.
- **y_c** is the current year.
- **c(p, n)** is the number of **citations of a paper on the n-th year** (e.g., c(10, 2018) = 114 means that the 10th paper had 114 citations in 2018).
- **f(x)** is an increasing function of **x**, in which **f(0)=1**. For example, an exponential function like f(x) = e^{x/10}
 can be used. In this case, citations in the first year have no weight added to them, and every citation after first year is multiplied by a positive weight.

##### **Note**
Year of the number of citations is subtracted from the publication year before being passed into f so that f(0) = 1 for the publication year, and the following years are passed into the function as 1, 2, 3, etc.

For example, using an exponential function like $f(x) = e^{x/10}$:
- f(0) = 1, meaning **citations in the first year after a paper's publication are not weighted**.
- f(7) \approx 2, meaning that **citations 7 years after a paper's publication are counted as 2 citations**, etc.

##### Example Usage & Example calculation
If **$$c(10, 2018) = 114$$** and the **publication year is 2015**, we calculate:

$$f(2018 - 2015) = f(3) \approx e^{3/10} \approx 1.3$$

Then, the weighted citations for this year are:

$$1.3 \times 114 = 148.2$$

Thus, 148.2 citations are summed up instead of 114. This function is applied to every citation amount for every year, for each paper.

**Conclusion**

If a dataset of an author with say 4000 citations is input and this sum returns 8000 weighted citations, 
we can tell that many of the authors paper's are **getting citations many years after being published**.
If the sum returns a number slightly larger, say 4500 we can tell that almost all of the author's publications
are very recent, or that most of the author's publications get no citations a few years after their publication.
Using a function s.t. $f(0)=1$ and $f\nearrow,  \forall  x\gt0$ means that the return of the function **will always
be greater than the input citations **. 
Using a different - decreasing function such as $f(x)=\exp(-x)^2$ we can have an index about the author's papers being citated (mostly) in their first year after publication.

## Using regression on given cites per paper dataset
 **find_line_of_best_fit( )** function takes a nested dictionary as input in  form 
``d = {paper_name: str:{year: str:citations_per_year: int}}``
and returns $$a, b, c$$ parameters for the regression function to be applied in the dataset, which is defined
in custom_function.
In current version based on research a function that can be used to fit the citations per year data is a gamma distribution function such as
$$ax^be^{(-cx)}$$ with a, b, c being float values to be found in the function.


### calculate_difference_from_mean( )
Given the paremeters $a_f, b_f, c_f, a_g, b_g, c_g$ of 
$$f(x)=g(x)= ax^be^{(-cx)}$$ (defined differently for different parameters) where $f(x)$ is the mean function of citations after x years and $g(x)$ is the function found using regression on an author's data using **find_line_of_best_fit( )**, this function returns the signed error from the mean calculated as:
$$\int_{0}^{b}f(x)-g(x)dx$$
where b tends to infinity. This integral returns the signed citations deviation from the mean.
In the code the integral is calculated as a  Riemman sum and b is set equal to 1000 for optimization reasons.

### plot_author_citations( )
Takes a nested dicitonary  in  form 
``d = {paper_name: str:{year: str:citations_per_year: int}}``
and plots the data as dots. The x-axis represents the years after a paper's publications while the y-axis
shows the citations of that paper, in x year.
