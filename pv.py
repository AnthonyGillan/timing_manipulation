# pv.py
# phase-locked phae vocoder implementation in Python
# slightly modified version of https://github.com/multivac61/pv

import sys
import numpy as np
from scipy.io import wavfile
from math import floor, ceil

# CONSTANTS
epsilon = sys.float_info.epsilon

class PhaseVocoder(object):
	"""docstring for PhaseVocoder"""
	def __init__(self, N=2**12, M=2**12, Rs=(2**12/8), w=np.hanning(2**12), alpha=1):
		super(PhaseVocoder, self).__init__()
		self.N	    = N		# FFT size
		self.M 	    = M		# Window size 
		self.Rs 	= Rs 	# Synthesis hop size
		self.alpha  = alpha	# Timestretch factor
		self.w      = w 	# Analysis/Synthesis window

	def speedx(self, sound_array, factor):
	    """ Multiplies the sound's speed by some `factor` """
	    indices = np.round( np.arange(0, len(sound_array), factor) )
	    indices = indices[indices < len(sound_array)].astype(int)
	    return sound_array[ indices.astype(int) ]

	def pitchshift(self, snd_array, n):
	    """ Changes the pitch of a sound by ``n`` semitones. """
	    factor = 2**(1.0 * n / 12.0)
	    stretched = self.timestretch(snd_array, 1.0/factor)
	    return self.speedx(stretched, factor)


	def timestretch(self, x, alpha):
		"""
		Perform timestretch of a factor alpha to signal x
		x: input signal, alpha: timestrech factor
		returns: a signal of length T*alpha
		"""

		# Analysis/Synthesis window function
		w = self.w; N = self.N; M = self.M
		hM1 = int(floor((M-1)/2.))
		hM2 = int(floor(M/2.))

		# Synthesis and analysis hop sizes
		Rs = self.Rs
		Ra = int(self.Rs / float(alpha))

		# AM scaling factor due to window sliding
		wscale = sum([i**2 for i in w]) / float(Rs)
		L = x.size
		L0 = int(x.size*alpha)

		# Get a prior approximation of the fundamental frequency
		if alpha != 1.0:
			A = np.fft.fft(w*x[0:N])
			B = np.fft.fft(w*x[Ra:Ra+N])
			Freq0 = B/A * abs(B/A)
			Freq0[Freq0 == 0] = epsilon
		else:
			Freq0 = 1

		if alpha == 1.0: 	# we can fully retrieve the input (within numerical errors)
			# Place input signal directly over half of window
			x = np.append(np.zeros(N+Rs), x)
			x = np.append(x, np.zeros(N+Rs))

			# Initialize output signal
			y = np.zeros(x.size)
		else:
			x = np.append(np.zeros(Rs), x)

			iteration_end = x.size - (Rs+N)
			num_iterations = ceil(iteration_end/float(Ra))

			num_samples_y = Rs*(num_iterations-1) + 1024

			# y = np.zeros(int((x.size)*alpha + (x.size/Ra)*alpha))
			y = np.zeros(int(num_samples_y))

		# Pointers and initializations
		p, pp = 0, 0
		pend = x.size - (Rs+N)
		Yold = epsilon

		i = 0
		while p < pend:
			i += 1
			# Spectra of two consecutive windows
			Xs = np.fft.fft(w*x[p:p+N])
			Xt = np.fft.fft(w*x[p+Rs:p+Rs+N])

			# Prohibit dividing by zero
			Xs[Xs == 0] = epsilon
			Xt[Xt == 0] = epsilon

			# inverse FFT and overlap-add
			if p > 0 :
				Y = Xt * (Yold / Xs) / abs(Yold / Xs)
			else:
				Y = Xt * Freq0

			Yold = Y
			Yold[Yold == 0] = epsilon

			y[pp:pp+N] += w*np.real(np.fft.ifft(Y))
			
			p = int(p+Ra)		# analysis hop
			pp += Rs			# synthesis hop

			# sys.stdout.write ("Percentage finishied: %d %% \r" % int(100.0*p/pend))
			sys.stdout.flush()

		y = y / wscale

		if self.alpha == 1.0:
			# retrieve input signal perfectly
			x = np.delete(x, range(N+Rs))
			x = np.delete(x, range(x.size-(N+Rs), x.size))
						
			y = np.delete(y, range(N))
			y = np.delete(y, range(y.size-(N+2*Rs), y.size))
		else:
			# retrieve input signal perfectly
			x = np.delete(x, range(Rs))

			y = np.delete(y, range(Rs))
			y = np.delete(y, range(L0, y.size))
							
		return y
