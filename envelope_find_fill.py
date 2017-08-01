import matplotlib.pyplot as plt
import random
import wave
from pylab import *
from pv import *
from gammatone_filters import *
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
from scipy.signal import butter, lfilter, hilbert

def white_noise(num_samples, mean=0, std=1):
	# int_samples=[]
	# for i in range (0, num_samples):
	# int_samples.append(int(samples[i]*32767))
	# return int_samples
	# samp_freq, audio=wavfile.read("sound_files/white_noise_22050.wav")
	# convert to a 1 channel jobby
	# audio=audio[0:num_samples]
	# audio_mono=[]
	# audio_mono=audio[:,1]
	# print audio_mono
	# print 'white noise length', len(audio_mono)
	# print 'number of samples', num_samples
	# audio_mono[num_samples:len(audio_mono)]=[]
	# print 'audio_mono', audio_mono
	samples=np.random.standard_normal(size=num_samples)
	this_max=np.amax(abs(samples))
	samples=samples/this_max
	return samples

def gammatone_create(samp_freq, num_freqs, lower_cutoff):
    # create a gammatone filterbank
    centre_freq_list=centre_freqs(samp_freq,num_freqs,lower_cutoff)
    coefs=make_erb_filters(samp_freq, centre_freq_list, width=1.0)
    return coefs

def louden(signal):
    this_max=np.amax(abs(signal))
    return signal/this_max
      
def control(samp_freq,snd,new_length, gammatone_coeffs, num_gammatone_bands):
##############################################################################################################################################
	# sound=louden(snd)       # normalise
	# noise=white_noise(snd.shape[0])  # create noise same length as sound
	# # noise=louden(noise)              # normalise
	# print 'noise', noise

	# filtered_signal_matrix=zeros((num_gammatone_bands, sound.shape[0])) # for gammatone decomposition of sound
	# filtered_noise_matrix=zeros((num_gammatone_bands, sound.shape[0]))  # for gammatone decomposition of noise

	# print "gammatone decompose word"    
	# filtered_signal_matrix=erb_filterbank(sound, gammatone_coeffs) # decompose the signal through the gammatone bank

	# print "gammatone decompose noise"   
	# filtered_noise_matrix=erb_filterbank(noise, gammatone_coeffs) # decompose the signal through the gammatone bank

	# # identify the envelope of each filter's output
	# # multiply the envelopes by the decomposed noise
	# # sum

	# envelopes=zeros((num_gammatone_bands, sound.shape[0]))
	# filled_envelope_matrix=zeros((num_gammatone_bands, sound.shape[0]))
	# s_control=zeros(sound.shape[0]) # list of length: how many samples there are in this gammavariated word

	# print "hilbert envelopes out and fill"
	# for i in range (0,num_gammatone_bands):
	#     # print 'band', (i+1)
	#     analytic_signal=hilbert(filtered_signal_matrix[i,:])        	# hilbert transfrom on each signal 
	#     envelopes[i,:]=np.abs(analytic_signal)   					    # abs value of analytic signal is signal envelope
	#     filled_envelope_matrix[i,:]=filtered_noise_matrix[i,:]*envelopes[i,:]     # multiply the envelopes with noise and sum

	# s_control=filled_envelope_matrix.sum(axis=0) # sum along columns
	# s_control=louden(s_control)

	# for i in range (0, sound.shape[0]):
	# 	s_control[i]=int((2**15)*s_control[i])
	
	# print 's_control dimensions are', s_control.shape
	# print "\n"
##############################################################################################################################################
	# sound=louden(snd)       # normalise
	sound = snd
	filtered_signal_matrix=zeros((num_gammatone_bands, sound.shape[0])) # for gammatone decomposition of sound
	filtered_signal_matrix=erb_filterbank(sound, gammatone_coeffs) 		# decompose the signal through the gammatone bank
	# identify the envelope of each filter's output
	# multiply the envelopes by the decomposed noise
	# sum
	band_envelopes=zeros((num_gammatone_bands, sound.shape[0]))
	main_envelope=zeros(sound.shape[0]) # list of length: how many samples there are in this gammavariated word

	# print "hilbert envelopes out and fill"
	for i in range (0,num_gammatone_bands):
	    analytic_signal=hilbert(filtered_signal_matrix[i,:])        	# hilbert transfrom on each signal 
	    band_envelopes[i,:]=np.abs(analytic_signal)   					    # abs value of analytic signal is signal envelope

	main_envelope=band_envelopes.sum(axis=0) # sum along columns

	# for i in range (0, sound.shape[0]):
	# 	main_envelope[i]=int((2**15)*main_envelope[i])

	return main_envelope                  
