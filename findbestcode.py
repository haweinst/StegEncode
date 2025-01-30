perr = .01
targeterrrate = .0000001
errrate = 1
ncands = 5
n=0
rate = 0
bestcand = (0,0)
minr = 0
minm = 0

r=0
m=0



import numpy as np
from scipy.stats import norm
from fractions import Fraction
from decimal import Decimal 




def choose(n,k):
    if k > n//2: k = n - k
    p = Fraction(1)
    for i in range(1,k+1):
        p *= Fraction(n - i + 1, i)
    return int(p)

#here if the perr is 1/10, p should be 10, if perr is 1/5, p should be 5. suitable for low values of p
def computeLerrRate(r, m, p):
    pcomp = p-1
    s=0
    for i in range(int((2**(m-r)-1)/2), 2**m+1): #sum over all uncorrectable errors
        s+=Decimal(choose(2**m, i))*Decimal(pcomp**(2**m-i))/Decimal(p**(2**m))
    return s

#approximation using normal distribution for high values of p (greater than .05)
def computeApproxLerrRate(r,m,p):
    q = 1-p
    nz=2**m
    print(str(r)+';'+str(m))
    s=1-norm(loc=p*nz, scale=np.sqrt(p*q*nz)).cdf(int((2**(m-r)-1)/2))
    print(s)
    return s

def makeRM(r, m):
    # print(str(r)+','+str(m))
    if r==0:
        return np.array([1]*(2**m))
    if m==r:
        return np.eye(2**m)

    else:
        a = makeRM(r-1, m-1)
        return np.block([
            [makeRM(r, m-1), makeRM(r, m-1)],
            [np.zeros(a.shape), a]
        ])


while n<ncands: 
    while errrate>targeterrrate:
        if(perr>.05):
            errrate = computeApproxLerrRate(r, m, perr)
        else:
            errrate = computeLerrRate(r,m,int(1/perr))
        # print(m)
        # print(errrate)
        m=m+1
    print(str(r)+';'+str(m))
    print(errrate)
    mat = makeRM(r,m)
    n=n+1
    if len(mat.shape)<2:
        if 1/mat.shape[0]>rate:
            bestcand=(r,m)
            minr = max(minr,r)
            rate = 1/mat.shape[0]
            # print(bestcand)
    else:
        if mat.shape[0]/mat.shape[1]>rate:
            bestcand=(r,m)
            minr = max(minr,r)
            rate = mat.shape[0]/mat.shape[1]
            # print(bestcand)
    errrate=1
    r=r+1
print('------------')
print('finished search')
print('best RM code parameters:')
mat = makeRM(bestcand[0],bestcand[1])
print(bestcand)
# print(mat.shape)
print('data rate:')
print(rate)
print('logical error rate:')
print("{:e}".format(computeLerrRate(bestcand[0], bestcand[1], int(1/perr))))