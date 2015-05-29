from numpy.fft import fft
from math import sqrt
import scipy
import scipy.stats

def erfc(x):
	return 1-scipy.special.erf(x)

def is_random(string):
	b = [int(i) for i in string]
	x = [2*i-1 for i in b]
	n  = len(x)
	X = fft(x)
	h = sqrt(2.995732274*n)
	N1 = len([ i for i in X[:n/2] if abs(i) < h])
	print N1
	N0 = 0.95*n/2
	d = (N1-N0)/sqrt(n*0.95*0.5*0.5)
	P = erfc(abs(d)/sqrt(2))
	
	if P < 0.01:
		return False
	else:
		return True
		

import urllib2
from time import sleep
for i in range(50):
        url = 'https://www.random.org/integers/?num=1000&min=0&max=1&col=1&base=10&format=plain&rnd=new'
        log = urllib2.urlopen(url).readlines()
        string = ''.join(map(lambda k: k[0],log))
        print is_random(string)
        sleep(5)
