# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:54:18 2021

@author: gordon
"""

def queenss(n):
    def placeQueens(k):
        return [[]] if k == 0 \
            else [[(k, column)] + queens 
        for queens in placeQueens(k - 1)
                                 for column in range(1, n + 1) 
                                     if isSafe((k, column), queens)]
    return placeQueens(n)

def isSafe(queen, queens):
    return all(not inCheck(queen, q) for q in queens)

def inCheck(q1, q2):
    return (q1[0] == q2[0] or 
            q1[1] == q2[1] or 
            abs(q1[0] - q2[0]) == abs(q1[1] - q2[1])) 

for qs in queenss(8):
    for q in qs:
        print(q, end="")
    print()