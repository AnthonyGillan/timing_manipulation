import numpy
import matplotlib.pyplot as plt

mean = 0
std = 1 
num_samples = 1000
samples = numpy.random.normal(mean, std, size=num_samples)

def main():

	plt.plot(samples)
	plt.show()

if __name__ == '__main__':
  main()