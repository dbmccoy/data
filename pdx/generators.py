def count_above_threshold(filename, threshold=6, column_number=2):
	with open(filename) as fi:
		headers = next(fi)
		return sum(1 for line in fi if float(line.split(",")[column_number]) > threshold)

##	CHALLENGE 1: Generate a small CSV and prove to yourself that the above works.


from random import normalvariate

def random_walk_until(reaches=10, standard_deviation=0.1):
	last_position = 0
	while abs(last_position) < until_reaches:
		new_position = last_position + normalvariate(0, standard_deviation)
		yield new_position
		last_position = new_position

# from context_managers import Fig
#
# def show_random_walk(**keywords):
#	with Fig(2, clear=True, log_x=False, log_y=False):
#		for i in range(7):
#			plot(tuple(random_walk_until(**keywords)), alpha=0.6)

## CHALLENGE 2: Count the number of steps until a random walk reaches some value of interest for the first time.
## CHALLENGE 3: Run that counter many times and generate a histogram of the distribution of travel times to your distance of interest.


def golden_ratio_min_finder(x_min, x_max):
	ratio = 0.5 * (5 ** 0.5 - 1)
	x_size = x_max - x_min
	x1 = x_max - ratio * x_size
	f1 = yield x1
	x2 = x_min + ratio * x_size
	f2 = yield x2
	while True:
		if f1 < f2:
			# print("switching x_max")
			x_max = x2
			x2 = x1
			f2 = f1
			x_size = x_max - x_min
			x1 = x_max - ratio * x_size
			f1 = yield x1
		else:
			# print("switching x_min")
			x_min = x1
			x1 = x2
			f1 = f2
			x_size = x_max - x_min
			x2 = x_min + ratio * x_size
			f2 = yield x2

from math import inf

def find_min(function, x_min, x_max, precision=1e-4, algorithm=golden_ratio_min_finder):
	finder = algorithm(x_min, x_max)
	# finder = algorithm(x_min, x_max, function)
	prior_x = inf
	current_x = finder.send(None)
	while abs(current_x - prior_x) > precision:
		prior_x = current_x
		current_x = finder.send(function(prior_x))
	return current_x

def simple_function(x):
	return 11.7 + x * (2.2 * x - 43.456790124)

def second_function(x):
    return x**3 - 6*x**2 + 4*x + 12

##	CHALLENGE 4: Try another, not-as-simple function to minimize
