import math
# Oklid Formula 
# $$d(A, B) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$


vector_A = [1,2]
vector_B = [5,6]

def Oklid(vector_A, vector_B):
    squareSum = 0
    count = len(vector_A)
    for i in range(count):
        a_i = vector_A[i]
        b_i = vector_B[i]
        diff = a_i - b_i
        square = diff**2
        squareSum = squareSum + square

    oklidDiff = math.sqrt(squareSum)
    return oklidDiff


def Manhattan(vector_A,vector_B):

    diff = 0
    sum = 0
    count = len(vector_A)
    for i in range(count):
        a_i  = vector_A[i]
        b_i = vector_B[i]
        
        sum = sum + abs(a_i - b_i)
    return sum



def Chebyshev(vector_A,vector_B):
    difs = []

    count = len(vector_A)
    for i in range(count):
        a_i = vector_A[i]
        b_i = vector_B[i]
        difs.append(abs(a_i - b_i))
    return max(difs)  


print(Chebyshev(vector_A,vector_B))