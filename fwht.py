
def rev(n,k=3):
	return int(bin(n)[2:].zfill(k)[::-1],2)
	
def reverse_order(a):
	perm = [rev(i) for i in range(len(a))]
	return [ a[i] for i in perm ]

def fwht(f):

	N=len(f) # a vektor hossza

	if N==1:
		return f

	# rekurzió 
	even=fwht( [ f[i] for i in range(0,N,2) ] ) #páros indexû tagok
	odd= fwht( [ f[i] for i in range(1,N,2) ] ) #páratlan indexû tagok

	M=N/2
	l=[ 1./M * even[k] + 1./M * odd[k] for k in range(M) ]
	r=[ 1./M * even[k] - 1./M * odd[k] for k in range(M) ]
	return l+r # összefűzzük a rekurzió során keletkező kisebb vektorokat


