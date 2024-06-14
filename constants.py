# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:23:38 2024

@author: marco
"""

MAX_ITER = 10

teta1 = 0.8
teta2 = 0.35  #SE LO ALZO, GradLagr DIMINUISCE, la DIMINUISCE, n_iter AUMENTA
teta3 = 0.7

EPS = [0.01, 0.01]

delta = 0.01

max_iter_penalita = 10**2

m = 8 #disuguaglianza
p = 0 #uguaglianza

DIM = 3

FILENAME_PROBLEM = "Funzione Test Prob1.txt"
FILENAME_STAMPE = "Risultati_PenSeq_MarcoCasalbore_es4.txt"