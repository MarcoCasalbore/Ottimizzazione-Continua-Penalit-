# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:27:23 2024

@author: marco
"""

import autograd.numpy as np
from autograd import grad

from constants import *





#PROBLEMA 1 


# m = 2
# p = 2

# def starting_point(n) :
#     return np.array([10.,10.,10.,10.])

# def dim() :
#     return 4

# def functionP(x, n): 
#     return -x[0]

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
#     g1 = -x[1]**2+x[0]**3
#     g2 = -x[0]**2-x[1]
    
#     return np.array([g1, g2])

# def vinU(x) :
#     h1=x[1]-x[0]**3-x[2]**2
#     h2=x[0]**2-x[1]-x[3]**2
#     return np.array([h1,h2])

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)








# m = 3
# p = 0

# def starting_point(n) :
#     return np.array([.5, .5, .5, .5])

# def dim() :
#     return 4

# def functionP(x, n): 
#     return x[0]**2 + float(0.5)*x[1]**2 + x[2]**2 + float(0.5)*x[3]**2 -x[0]*x[2] + x[2]*x[3]-x[0]-3*x[1]+x[2]-x[3]

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
#     g1 = x[0]+2*x[1]+x[2]-x[3]-5
#     g2 = 3*x[0]+x[1]+2*x[2]-x[3]-4
#     g3=-x[1]+4*x[2] + float(1.5)
#     return np.array([g1, g2,g3])

# def vinU(x) :
    
#     return 0

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)

#FILE 6 PROB 100
# m = 4
# p = 0

# def starting_point(n) :
#     return np.array([1.,2., 0., 4., 0.,1.,1.])

# def dim() :
#     return 7

# def functionP(x, n): 
#     return (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 +10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5]-8*x[6]


# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
#     g1 = 2*x[0]**2 + 3*x[1]**4 +x[2]+4*x[3]**2 + 5*x[4] - 127
#     g2 = 7*x[0] + 3*x[1] +10*x[2]**2 + x[3]-x[4]-282
#     g3= 23*x[0]+x[1]**2+6*x[5]**2 +8*x[6]-196
#     g4= 4*x[0]**2 +x[1]**2 -3*x[0]*x[1] - 2* x[2]**2 + 5*x[5] -11*x[6]
#     return np.array([g1, g2,g3,g4])
    
   

# def vinU(x) :
    
#     return 0

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu,n) 







#FILE 6 PROB 77
# m = 0
# p = 2

# def starting_point(n) :
#     return np.array([2.,2., 2., 2., 2.])

# def dim() :
#     return 5

# def functionP(x, n): 
#     return (x[0]-1)**2 + (x[0]-x[1])**2 + (x[2]-1)**2 +(x[3]-1)**4 + (x[4]-1)**6
# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
    
#     return 0

# def vinU(x) :
#     h1 = (x[0]**2)*x[3] + np.sin(x[3]-x[4]) - 2*np.sqrt(2)
#     h2 = x[1]+ (x[2]**4)*x[3]**2 - 8 - np.sqrt(2)
#     return np.array([h1, h2])

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu,n) 





# m = 2
# p = 2

# def starting_point(n) :
#     return np.array([10., 10., 10., 10.])

# def dim() :
#     return 4

# def functionP(x, n): 
#     return -x[0]

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
#     g1 = x[0]**3 - x[1]
#     g2 = x[1] - x[0]**2
#     return np.array([g1, g2])

# def vinU(x) :
#     h1 = x[1] - x[0]**3 - x[2]**2
#     h2 = x[0]**2 - x[2] - x[3]**2
#     return np.array([h1, h2])

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



### FILE4 - 2 ###

# m = 0
# p = 2

# def dim() :
#     return 3

# def starting_point(n) :
#     return np.array([0., 0., 0.])

# def functionP(x, n):
#     return 4*x[0]**2 + 2*x[1]**2 + 2*x[2]**2 - 33*x[0] + 16*x[1] - 24*x[2]

# def vinD(x) :
#     #g1
#     #gm
#     return np.array([])

# def vinU(x) :
#     h1 = 3*x[0] - 2*x[1]**2 - 7
#     h2 = 4*x[0] - x[2]**2 - 11
#     #hp
#     return np.array([h1, h2])

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     for i in range(m) :
#         L += vinD(x)[i] * la[i]
#     #print("Lla", Lla)
#     for j in range(p) :
#         L += vinU(x)[j] * mu[j]
#     #print("Lmu", Lmu)
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



### FILE5 - 2 ###

# m = 9
# p = 1

# def dim() :
#     return 4

# def starting_point(n) :
#     return np.array([1., 5., 5., 1.])

# def functionP(x, n):
#     return x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]

# def vinD(x) :
#     g1 = - x[0]*x[1]*x[2]*x[3] + 25
#     g2 = 1 - x[0]
#     g3 = x[0] - 5
#     g4 = 1 - x[1]
#     g5 = x[1] - 5
#     g6 = 1 - x[2]
#     g7 = x[2] - 5
#     g8 = 1 - x[3]
#     g9 = x[3] - 5
#     #gm
#     return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

# def vinU(x) :
#     h1 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40
#     #hp
#     return np.array([h1])

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     for i in range(m) :
#         L += vinD(x)[i] * la[i]
#     #print("Lla", Lla)
#     for j in range(p) :
#         L += vinU(x)[j] * mu[j]
#     #print("Lmu", Lmu)
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)


### FILE1 - 4 ###

# m = 3
# p = 0

# def dim() :
#     return 2

# def starting_point(n) :
#     return np.array([-2., 1.])

# def functionP(x, n):
#     return 100*(x[1]- x[0]**2)**2 +(1-x[0])**2

# def vinD(x) :
#     g1 = -x[0]*x[1] + 1
#     g2 = -x[0] - x[1]**2
#     g3 = x[0] - 0.5
    
#     return np.array([g1, g2, g3])

# def vinU(x) :
#     return np.array([])

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     for i in range(m) :
#         L += vinD(x)[i] * la[i]
#     #print("Lla", Lla)
#     for j in range(p) :
#         L += vinU(x)[j] * mu[j]
#     #print("Lmu", Lmu)
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)


### FILE2 - 4 ###

# m = 6
# p = 0

# def dim() :
#     return 2

# def starting_point(n) :
#     return np.array([20.1 , 5.84])

# def functionP(x, n):
#     return (x[0]-10)**3 + (x[1]-20)**3

# def vinD(x) :
#     g1 = 100 - (x[0]-5)**2 - (x[1]-5)**2
#     g2 = (x[1]-5)**2 + (x[0]-6)**2 - 82.81
#     g3 = x[0] - 100
#     g4 = 13 - x[0]
#     g5 = x[1] - 100
#     g6 = -x[1]
#     return np.array([g1,g2,g3,g4,g5,g6])

# def vinU(x) :
#     return np.array([])

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     for i in range(m) :
#         L += vinD(x)[i] * la[i]
#     #print("Lla", Lla)
#     for j in range(p) :
#         L += vinU(x)[j] * mu[j]
#     #print("Lmu", Lmu)
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



### FILE1 - 1 ###

# m = 1
# p = 0

# def dim() :
#     return 2

# def starting_point(n) :
#     return np.array([-4.9,1.])

# def functionP(x, n):
#     return (x[0]-5)**2 + (x[1])**2 - 25

# def vinD(x) :
#     g1 = x[0]**2 - x[1]
#     return np.array([g1])

# def vinU(x) :
#     return np.array([])

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     for i in range(m) :
#         L += vinD(x)[i] * la[i]
#     #print("Lla", Lla)
#     for j in range(p) :
#         L += vinU(x)[j] * mu[j]
#     #print("Lmu", Lmu)
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)


#FILE3 ES (ULTIMO)

# m = 10
# p = 3

# def starting_point(n) :
#     return np.array([2., 2., 2., 2.,2.])

# def dim() :
#     return 5

# def functionP(x, n): 
#     return (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2

# def vincoliP(x): #funzione di penalità
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) :
#     g1 = x[0]-10
#     g2 = x[1]-10
#     g3 = x[2]-10
#     g4 = x[3]-10
#     g5 = x[4]-10
#     g6 = -x[0]-10
#     g7 = -x[1]-10
#     g8 = -x[2]-10
#     g9 =-x[3]-10
#     g10=-x[4]-10
    

#     return np.array([g1, g2,g3,g4,g5,g6,g7,g8,g9,g10])

# def vinU(x) :
#     h1 = x[0]+3*x[1]
#     h2 = x[2]+x[3]-2*x[4]
#     h3=x[1]-x[4]
#     return np.array([h1, h2,h3])

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



#FILE 4 ES 60

# m = 6
# p = 1

# def starting_point(n) :
#     return np.array([2., 2., 2.])

# def dim() :
#     return 3

# def functionP(x, n): 
#     return (x[0]-1)**2 + (x[0]-x[1])**2 + (x[1]-x[2])**4 

# def vincoliP(x): #funzione di penalità NON CAMBIARE
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità NON CAMBIARE
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) : #imnore uguale di zero
#     g1 = x[0]-10
#     g2 = x[1]-10
#     g3 = x[2]-10
#     g4 = -x[0]-10
#     g5 =-x[1]-10
#     g6=-x[2]-10
    

#     return np.array([g1, g2,g3,g4,g5,g6])

# def vinU(x) : #quanto non ci sono mettiamo return 0
#     h1 = x[0]*(1+x[1]**2)+x[2]**4-4-3*(np.sqrt(2))
   
#     return np.array([h1])

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



    ######################################### PROBLEMA PER L'ESAME #########################################################



# m = 5
# p = 0

# def starting_point(n) :
#     return np.array([3., 1.])

# def dim() :
#     return 2

# def functionP(x, n): 
#     return x[0]**2 + x[1]**2

# def vincoliP(x): #funzione di penalità NON CAMBIARE
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità NON CAMBIARE
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) : #imnore uguale di zero
#     g1 = -x[0] - x[1] + 1
#     g2 = -x[0]**2 - x[1]**2 +1
#     g3 = -9*x[0]**2 - x[1]**2 +9
#     g4 = x[1] - x[0]**2
#     g5 = x[0] - x[1]**2
#     # g6=-x[2]-10
    

#     return np.array([g1, g2, g3, g4, g5])

# def vinU(x) : #quanto non ci sono mettiamo return 0
#     # h1 = x[1] - (x[0]**3) - (x[2]**2)
#     # h2 = (x[0]**2) - x[1] - (x[3]**2)
   
#     #return np.array([h1, h2])
#     return 0

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



# m = 2
# p = 0

# def starting_point(n) :
#     return np.array([0., 0.])

# def dim() :
#     return 2

# def functionP(x, n): 
#     return x[1]

# def vincoliP(x): #funzione di penalità NON CAMBIARE
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità NON CAMBIARE
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) : #imnore uguale di zero
#     g1 = 2*x[0]**2 - x[1]**3 - x[1]
#     g2 = 2*((1-x[0])**2) - (1-x[0])**3 -x[1]
#     # g3 = -9*x[0]**2 - x[1]**2 +9
#     # g4 = x[1] - x[0]**2
#     # g5 = x[0] - x[1]**2
#     # # g6=-x[2]-10
    

#     return np.array([g1, g2])

# def vinU(x) : #quanto non ci sono mettiamo return 0
#     # h1 = x[1] - (x[0]**3) - (x[2]**2)
#     # h2 = (x[0]**2) - x[1] - (x[3]**2)
   
#     #return np.array([h1, h2])
#     return 0

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



# m = 3
# p = 0

# def starting_point(n) :
#     return np.array([-0.1, -0.9])

# def dim() :
#     return 2

# def functionP(x, n): 
#     return -x[1]

# def vincoliP(x): #funzione di penalità NON CAMBIARE
#     return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

# def P(x, eps, n): #funzione di penalità NON CAMBIARE
#     return functionP(x, n) + (1 / eps) * vincoliP(x)
    
# def vinD(x) : #imnore uguale di zero
#     g1 = 2*x[1] - x[0] -1
#     g2 = -(x[0]**2) -(x[1]**2)
#     g3 = -1 + x[0]**2 + x[1]**2
#     # g4 = x[1] - x[0]**2
#     # g5 = x[0] - x[1]**2
#     # # g6=-x[2]-10
    

#     return np.array([g1, g2, g3])

# def vinU(x) : #quanto non ci sono mettiamo return 0
#     # h1 = x[1] - (x[0]**3) - (x[2]**2)
#     # h2 = (x[0]**2) - x[1] - (x[3]**2)
   
#     #return np.array([h1, h2])
#     return 0

# def Lagr(x, la, mu, n) :
#     L = functionP(x, n)
#     constraintsD = vinD(x)
#     constraintsU = vinU(x)
#     for i in range(m) :
#         L += constraintsD[i] * la[i]
        
#     for j in range(p) :
#         L += constraintsU[j] * mu[j]
#     return L

# def nablaLagr(Lagr, x, la, mu, n) :
#     return grad(Lagr)(x, la, mu, n)



m = 8
p = 0

def starting_point(n) :
    return np.array([22.3, 0.5, 125])

def dim() :
    return 3

def functionP(x, n): 
    return (-0.0201*x[0]**4 * x[1] * x[2]**2) * 10**(-7)

def vincoliP(x): #funzione di penalità NON CAMBIARE
    return (np.sum((np.maximum(0.0, vinD(x)))**2) + (np.sum(vinU(x)**2)))

def P(x, eps, n): #funzione di penalità NON CAMBIARE
    return functionP(x, n) + (1 / eps) * vincoliP(x)
    
def vinD(x) : #imnore uguale di zero
    g1 = -675 + x[0]**2 * x[1]
    g2 = -0.419 + 10**(-7) *x[0]**2 * x[2]**2
    g3 = -x[0]
    g4 = -x[1]
    g5 = -x[2]
    g6 = x[0] -36
    g7 = x[1] - 5
    g8 = x[2] - 125
    

    return np.array([g1, g2, g3, g4, g5, g6, g7, g8])

def vinU(x) : #quanto non ci sono mettiamo return 0
    # h1 = x[1] - (x[0]**3) - (x[2]**2)
    # h2 = (x[0]**2) - x[1] - (x[3]**2)
   
    #return np.array([h1, h2])
    return 0

def Lagr(x, la, mu, n) :
    L = functionP(x, n)
    constraintsD = vinD(x)
    constraintsU = vinU(x)
    for i in range(m) :
        L += constraintsD[i] * la[i]
        
    for j in range(p) :
        L += constraintsU[j] * mu[j]
    return L

def nablaLagr(Lagr, x, la, mu, n) :
    return grad(Lagr)(x, la, mu, n)



#####################################################################################################################################

# if __name__ == "__main__":
#     from scipy.optimize import minimize, dual_annealing, basinhopping, differential_evolution

#     def functionProblemScipy(x) :
#         f = (np.exp(x[0]) - x[1])**4 + 100*((x[1] - x[2])**6) + (np.tan(x[2] - x[3]))**4 + x[0]**8
#         return f
    
#     n = 4

#     x0 = starting_point(4)
#     # bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
#     bounds = [(-1, 1)  for i in range(n)]
#     print(bounds)

    # ===========  LOCAL MINIMIZER  =============
    # result = minimize(functionProblemScipy, x0)

    # ===========  GLOBAL MINIMIZER  =============

    #COME IL SIMULATED ANNEALING
    # result = dual_annealing(functionProblemScipy, bounds)



    # BASINSHOPPING
    # Definisci il minimizzatore locale
    # minimizer_kwargs = {"method": "BFGS"}
    # result = basinhopping(functionProblemScipy, x0, minimizer_kwargs=minimizer_kwargs)



    # DIFFERENTIAL_EVOLUTION(sembra essere il migliore)
    # result = differential_evolution(functionProblemScipy, bounds)

    # print("Optimal value:", result.fun)
    # print("Optimal point:", result.x)