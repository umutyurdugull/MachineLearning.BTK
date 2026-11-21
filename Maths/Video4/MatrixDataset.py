import numpy as np

# Example customer dataset 
dataset = [
    [1,2500,8,12,85],
    [2,800,2,5,40],
    [3,1200,4,7,55]
    ]

# customer id, monthly spent, visit frequency, product freq. , loyalty points.
# In machine learning they are property and every row is property vector 
# example 

vector1 = [1,2500,8,12,85]  


# ax = b  
#  | 2    180 | x1    | 3 |
#  | 4    300 | x2 =  | 8 |

# Ax = b  --> x = A^-1b
A = np.array([
    [2, 180],
    [4, 300]
])

b = np.array([3,8])

A = np.linalg.inv(A)

x = A @ b
print(x)


# What if we have 5,450 and Y ? How can we find Y ?