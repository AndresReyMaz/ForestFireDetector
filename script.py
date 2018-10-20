import itertools

with open('clasificacion1.txt') as f:
	lines1 = [line.rstrip('\n') for line in f]

with open('clasificacion1.txt') as f:
	lines2 = [line.rstrip('\n') for line in f]

for (a,b) in zip(lines1, lines2):
	print(a + "," + b)