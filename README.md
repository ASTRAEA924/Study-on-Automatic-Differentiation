# Simple Automatic Differentiation

## Running Example
The following example generates Table 2 in [Automatic Differentiation in Machine Learning: A Survey (Baydin et al., 2018)](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf), which calculates the partial derivative with respect to the first variable $x_1$.
We consider $y=\log(x_1)+x_1x_2-\sin(x_2)$ with $(x_1,x_2)=(2,5)$.
```python
from simpleautodiff import *

Node.verbose = True

# create root nodes
x1 = Node(2)
x2 = Node(5)

# create computational graph and evaluate function value
y = sub(add(log(x1), mul(x1, x2)), sin(x2))
# perform reverse-mode autodiff


reverse(y)


print("\nResults:")
print(f"dy/dx1 = {x1.gradient:.6f}")
print(f"dy/dx2 = {x2.gradient:.6f}")

Output
```
It first creates the computational graph 

```
x1 = input[]            = 2       
x2 = input[]            = 5       
v1 = log['x1']          = 0.693   
v2 = mul['x1', 'x2']    = 10      
v3 = add['v1', 'v2']    = 10.693  
v4 = sin['x2']          = -0.959  
v5 = sub['v3', 'v4']    = 11.652  
```
Then, the code performs reverse-mode automatic differentiation 
```
dv5/dv3 += (dv5/dv5) * (dv5/dv3) = 0.000 + 1.000 * 1.000 -> 1.000
dv5/dv4 += (dv5/dv5) * (dv5/dv4) = 0.000 + 1.000 * -1.000 -> -1.000
dv5/dx2 += (dv5/dv4) * (dv4/dx2) = 0.000 + -1.000 * 0.284 -> -0.284
dv5/dv1 += (dv5/dv3) * (dv3/dv1) = 0.000 + 1.000 * 1.000 -> 1.000
dv5/dv2 += (dv5/dv3) * (dv3/dv2) = 0.000 + 1.000 * 1.000 -> 1.000
dv5/dx1 += (dv5/dv2) * (dv2/dx1) = 0.000 + 1.000 * 5.000 -> 5.000
dv5/dx2 += (dv5/dv2) * (dv2/dx2) = -0.284 + 1.000 * 2.000 -> 1.716
dv5/dx1 += (dv5/dv1) * (dv1/dx1) = 5.000 + 1.000 * 0.500 -> 5.500

```
Then, the code print the final resault 
```

Results:
dy/dx1 = 5.500000
dy/dx2 = 1.716338
