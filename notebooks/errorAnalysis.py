def factorial(n):
    fact=1 # keep track of the factorial
    for i in range(n):
        fact = fact*(i+1) #+1 because the first i is zero
    return fact
    
def squareCube(x):
  #return a tuple
  return x*x, x*x*x 