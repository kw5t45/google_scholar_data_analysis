# Google Scholar "API"

## Data fetching
This library scrapes data from google scholar using scholarly library.
Creating an *Author object* using only their Unique Google Scholar ID, using Author class methods
we can save their **Profile** data, such as **Author name, interests, co-authors, citations**
and other **indexes** that appear in their google scholar profile.
Author's **paper data** can also be saved, which saves important data for each one of the author's papers, such as **title, co-authors, number of citations, ID's publisher, pages, (all) authors, citations per year** and more.
All of that data can be saved and retrieved in a JSON format file using methods in the code.



## Analysis
### Weighted Citations
Using **weight_citations_based_on_function_of_time** we can "weigh" all of the citations of an author 
based on how **recent** the citations are, compared to how **old** the paper is, using:
$$\sum_{p=1}^{papers}\sum_{n=y_0}^{y_c}f(n-y_n)c(p, n)$$
Where *papers* is the number of all authors publications, $y_c$ is the current year $c(p, n)$ is the number of **citations of paper on n-th year** (e.g. $c(10, 2018) = 114$ means that the 10th paper had 114 citations in 2018),  and $f(x)$ is an increasing function  of x, in which $f(0)=1$, so the citations in the first year have no weight added to them, e.g. $\exp(x/10)$. Year of the number of citations is subtracted from publication year before
being passed into $f$ so that $f(0)=1$ for the publication year, and the following years are passed into the 
function as 1, 2, 3 etc.
For example, using an exponential function like $e^{x/10}$ we have:
$f(0)=1$, meaning **citations in the first year after a paper's publication are not weighted**,
$f(7)=2$ (aprox.), meaning that **citations 7 years after a papers publication are counted as 2 citations**, etc.
In practice, if $c(10, 2018) = 114$ and **publication year is 2015** we multiply by $f(2018 - 2015) = f(3) \simeq1.3*114=148.2$, meaning that 148.2 citations are summed up instead of 114. This function is applied to every citation amount on every year, for each paper.

**Conclusion**

If we input a dataset of an author with say 4000 citations and this sum returns 8000 weighted citations, 
we can tell that many of the authors paper's are **getting citations many years after being published**.
If the sum returns a number slightly larger, say 4500 we can tell that almost all of the author's publications
are very recent, or that most of the author's publications get no citations a few years after their publication.
Using a function s.t. $f(0)=1$ and $f\nearrow,  \forall  x\gt0$ means that the return of the function **will always
be greater than the input citations **. 
Using a different - decreasing function such as $f(x)=\exp(-x)^2$ we can have an index about the author's papers being citated (mostly) in their first year after publication.
