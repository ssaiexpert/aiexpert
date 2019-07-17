import numpy as np

# Return a single sample sampled from a negative binomial NB(r,p) distribution.
def sample_nb(r, p):
	# Your code here
	neg_count = 0
	pos_count = 0

	while neg_count < r:
		# sample from a bernoulli distribution
		flip = np.random.negative_binomial(r, p, 1)
		# check if heads or tails and keep track
		if flip == 0:
			neg_count += 1
		elif flip == 1:
			pos_count += 1
	return pos_count

# Execute monte carlo to determine E[x] where x = NB(r,p). Return the resulting value.
def montecarlo(r, p, N):
	# Your code here
	sum = 0
	# sample N times
	for n in range(N):
		sum += sample_nb(r, p)

	# divide by N
	sum /= N
	return sum

if __name__ == "__main__":
	# Use this section to execute and check your implementation.
	# This section is not used when grading.
	N = 10000
	print("Calculate the mean of NB(3, 0.7).")
	mean = montecarlo(3, 0.7, N)
	print("Mean value approximated with {0} samples: {1}".format(N, mean))