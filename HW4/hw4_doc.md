# HW 4 Jacob Reiss & Michael Zeolla

## Problem 1
To represent the operation as a fully connected layer, we can pad W with 0s making it a circulant matrix:
$W = \begin{bmatrix} 1 \ 3 \ {-1} \ 2 \ 0 \ 0 \ 0 \ 0 \ 0 \\ 0 \ 1 \ 3 \ {-1} \ 2 \ 0 \ 0 \ 0 \ 0 \\ 0 \ 0 \ 1 \ 3 \ {-1} \ 2 \ 0 \ 0 \ 0 \\ 0 \ 0 \ 0 \ 1 \ 3 \ {-1} \ 2 \ 0 \ 0 \\ 0 \ 0 \ 0 \ 0 \ 1 \ 3 \ {-1} \ 2 \ 0 \\ 0 \ 0 \ 0 \ 0 \ 0 \ 1 \ 3 \ {-1} \ 2\end{bmatrix}$

