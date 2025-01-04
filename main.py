from analysis import *
from plots import plot_co_author_graph, get_co_author_graph_pairs
# TO DO *************************************

# test

# add src (and core?) folders
# check for errors in dynamic
# make error case for dicitonary type hint
# verbrose parameter, suppress output
# ULTIMATE: FIND ERDOS NUMBER FUNCTION
# get author data from json?
# parameterize co author data only
# time proccess in some functions
# sort coauthors
# clip d2 coauthors
# limit at publications
# pass kwargs in plot
# input pause?
# save plots

# QUESTIONS FOR DASKALO *************
# vebrose
# core? src?
# testing genikotera
# custom type hint? eg dict[dict[list[.....]]]]
# PAPERS!!!!


# few papers jP1qgO4AAAAJ
# papak id O9d4j7oAAAAJ

#checkpoint_save_author_and_coauthors_in_tree('O9d4j7oAAAAJ', clip=400)
plot_co_author_graph(get_co_author_graph_pairs('rIug0ugAAAAJ'))
