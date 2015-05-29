import numpy as np
from numpy.fft import fft,ifft

def mult(p,q):
	p_len = len(p)+len(q)-1
	q_len = len(q) + len(p)-1
	
	power = max(q_len,p_len)
	if  not 3>bin(power).rfind('1'):
		power = int(np.ceil(np.log2(power)))
		
	p = p + [0 for i in range(2**power-len(p))]
	q = q + [0 for i in range(2**power-len(q))]
	
	return ifft(fft(p)*fft(q))

p = [0,0,0,1,4,0,10]
q = [5,12,-3,1,0,0,0,1]

print 'p:', p
print 'q:', q

print 'p*q:'
for i in mult(p,q):
	print i
