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
after analyzing a program, choose a set of tilings which is optimal for communication
costs


