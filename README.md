# TilingSolver

This project was created to aid in making tiling decisions for distributed
matrix applications, specifically with an eye for applications in the Phylanx 
(https://github.com/stellar-group/phylanx) project.

By tiling we mean the decisions regarding how to distribute data across some
number of computational nodes, for example, in a cluster or supercomputer.

For example, if we wish to distribute data for an add operation across four
localities, we need the subsections (tiles) of the matrices to be of the proper
type for the communication costs to be minimized.

<p align="center">
  <img width="426" height="426" src="/images/a_row_b_row_c_col.png">
</p>

In the first example here, it is not.

<p align="center">
  <img width="426" height="426" src="/images/a_row_b_row_c_row.png">
</p>

In the second example, it is. Broadly speaking, the goal of the TilingSolver is to,
after analyzing a program, choose a set of tilings and algorithms which is optimal
for communication costs. 

This set of tilings and algorithms is discovered in one of three main ways
(with another trivial implementation as well).

1. We iterate exhaustively through all possibilities, and calculate the cost
of each option, choosing the one with the lowest cost.

2. We iterate through all possibilities for algorithms, choosing the set of
algorithms which we can find the best solution for using a non-exhaustive greedy
search

3. We choose the algorithms with the tightest performance bounds, and the tilings
obtained from the same non-exhaustive greedy search used in option #2

We expect the viability of each of these 3 options to change as the number of 
variables and operations increase, so that the options with higher number become
preferable as the number of involved variables and operations increase. This is due
to each one's asymptotic complexity. 
