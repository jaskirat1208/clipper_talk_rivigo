import random
for i in xrange(1,1000):
	ls = []
	for j in xrange(1,786):
		ls.append(random.randint(1,255))
	ls = ','.join(map(str, ls))
	print ls 
