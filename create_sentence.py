import sys, getopt
import wave
import struct
from envelope_find_fill import *
from pv import *
from spectrogram import *
from spatial_filters import *
from pylab import *
from random import gammavariate
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, hilbert, stft, istft, spectrogram
from scipy import stats
import pywt
import numpy as np
from numpy.lib import stride_tricks
import peakutils
import matplotlib.pyplot as plt

class Sentence:

    def __init__(self, sentence_file, new_length):
        text_file = open("sentences/"+sentence_file+".txt", "r") # open file, read mode
        lines = text_file.readlines()        # returns a list of lines in file, these files contain one word per line
        lines =[s.strip() for s in lines]    # trims trailing and leading whitespace
        text_file.close()        
        self.words = []                        # instance variable for words (audio)
        self.word_names = []                   # instance variable for words (text)
        word_freqs = []                        # instance variable for sample rates
        self.t_lengths = []                    # instance variable for time lengths of words
        self.samples_in_sentence = 0
        self.new_length = new_length
        print "\n"

        for word_name in lines:               # iterate through words (text) as only one word per line
            self.word_names.append(word_name) # append words (text) to list

            audio_seg_word = AudioSegment.from_file("words_audio/"+sentence_file+"/"+word_name+".wav") 	# open as audiosegment to strip silence 
            word_freq, w = wavfile.read("words_audio/"+sentence_file+"/"+word_name+".wav") 		   	   	# do this to get sample rate

            start_trim_time = self.detect_leading_silence(audio_seg_word)						
            end_trim_time = self.detect_leading_silence(audio_seg_word.reverse())
            duration_time = len(audio_seg_word)

            # convert times to seconds (/1000) then to amount of samples
            start_trim = int(word_freq * (double(start_trim_time) / 1000))
            end_trim = int(word_freq * (double(end_trim_time) / 1000))
            duration = int(word_freq * (double(duration_time) / 1000))

            word = w[start_trim:duration - end_trim]				     # perform the trim
            
            self.words.append(word)                                      # append word (audio) to 'words'
            word_freqs.append(word_freq)                                 # append sample rate of .wav to word_freqs
            self.t_lengths.append(double(word.shape[0]) / word_freq)       # (number of rows in sample data ie samples)/(sample rate) = time length of word
            self.samples_in_sentence += word.shape[0]
            print word.shape,word_freq,self.t_lengths[-1],word_name      # number of samples, sample rate, time legth[last in list], word (text)

        self.freq = word_freqs[0]                                          # first sample rate of word
        for freq in word_freqs:                                          # check if sample rates are the same for all words
            if freq != self.freq:
                print "not all the freqs the same f/p"

        self.length = len(self.word_names)                                 # amount of words in sentence
        print 'number of words in sentence is', self.length
        print 'number of samples in sentence is', self.samples_in_sentence
        print "\n"

    def detect_leading_silence(self, sound, silence_threshold=-55.0, chunk_size=5):
        # sound is a pydub.AudioSegment
        # silence_threshold in dB
        # chunk_size in ms
        # iterate over chunks until you find the first one with sound
        trim_ms = 0 # ms
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold:
            trim_ms += chunk_size
        return trim_ms

class Control_sentence:

	def __init__(self,words,freq,word_names):
	    self.words = words                        # instance variable for words (audio)
	    self.word_names = word_names              # instance variable for words (text)
	    self.t_lengths = []                       # instance variable for time lengths of words

	    for word in self.words:                 # iterate through words (audio) 
	        self.t_lengths.append(double(word.shape[0]) / freq) # (samples in audio clip)/(sample rate) = time length of word

	    self.freq = freq                                        # instance variable for sample rate
	    self.length = len(self.word_names)                      # amount of words in sentence

def s_concatenate(sentence, word_length, freq, sentence_number):

    s = []
    silence_lenght = 0.2 - (0.2*word_length)			# shorter the longer the words are stretched to
    if silence_lenght<0:
    	silence_lenght=0

    silence_samples = zeros(int(silence_lenght*freq))

    for word in sentence.words:
        s = concatenate([s, word])                   # put words (audio) into array s
        # s = concatenate([s, silence_samples])		 # put silences between the words

    s_int16 = zeros((s.shape[0]), dtype='int16')     # fill 0's (int16) in array with same number of rows as samples in all words here

    for i in range(0, s.shape[0]):                   # fill s with all int16 samples from 0 to #rows
        s_int16[i] = s[i]

    wavfile.write(""+sentence_number+".wav", sentence.freq, s_int16)   # (filename, sample rate, data array of int16 samples in sentence)
    
    return s_int16

def stretch_simple(sound, f, window_size=2**10, h=2**10/4):
    # Stretches the sound by a factor 'f'
    window_size = int(window_size)
    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(sound) / f) + window_size)

    for i in np.arange(0, int(len(sound)) - int((window_size + h)), int(h*f)):

	    # two potentially overlapping subarrays
	    a1 = sound[i: i + window_size]
	    a2 = sound[i + h: i + window_size + h]

	    # resynchronize the second array on the first
	    s1 =  np.fft.fft(hanning_window * a1)
	    s2 =  np.fft.fft(hanning_window * a2)
	    phase = (phase + np.angle(s2/s1)) % 2*np.pi
	    a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

	    # add to result
	    i2 = int(i/f)
	    result[i2 : i2 + window_size] += np.real(hanning_window * a2_rephased)

    return result.astype('int16')

def stretch(word, stretch_factor):
    N = 2**10					# Number of channels
    M = 2**10					# Size of window
    w = np.hanning(M-1)			# Type of Window (Hanning)
    w = np.append(w, [0])		# Make window symmetric about (M-1)/2
    Os = 8.0					# Synthesis hop factor
    Rs = int(N / Os)			# Synthesis hop size
    alpha = stretch_factor

    pv = PhaseVocoder(N, M, Rs, w, alpha)
    new_w = pv.timestretch(word, alpha)  		  # stretch word by factor alpha

    return new_w.astype('int16')

def fades(word, percentage_fade):
	fade_length_samples = int(word.size * percentage_fade)
	fade_factor = 0
	fade_increment = 1/float(fade_length_samples)

	for i in range(0, fade_length_samples): # fade from start
		fade_factor += fade_increment
		word[i] *= fade_factor
		word[i] = int(word[i])

	fade_factor = 1
	for i in range(word.size-fade_length_samples, word.size): # fade to end
		fade_factor -= fade_increment
		word[i] *= fade_factor
		word[i] = int(word[i])

	return word

def main(argv):

	base_l = 0.4                                            # base length
	new_l_av = 0.2                                          # new average length 
	k = 0.1                                                 # shape of distribution
	theta = new_l_av/k                                      # scale of distribution

	sentence_file = ''
	new_length = 1.0

	try:                                                   # runs in entirety unless exception
		opts, args = getopt.getopt(argv,"hb:e:n:", ["sfile=", "stretch=", "newl=" ])
	except getopt.GetoptError:                             # raised when an unrecognized option is found in the argument list or when an option requiring an argument is given none
		print 'load_sentence.py -i <sentence_file> -s <new_length(float)>'         # <input file>
		sys.exit(2)                                         # exit from python -> arg is an int (2) giving the exit status
	for opt, arg in opts:                                  # cycle through options
		if opt == '-h':
			print 'load_sentence.py -b <start sentence number> -e <end sentence number> -n <new word length (s) for words other than middle>'
			sys.exit()                                    # exit from python
		elif opt in ("-b", "--begin"):                    # the sentence file to begin from
			start_file = float(arg)                       
			start_file = int(start_file)         
		elif opt in ("-e", "--end"):                      # the sentence file to end at
			stop_file = float(arg)                        
			stop_file = int(stop_file)
		elif opt in ("-n", "--newl"):                  	  # the length in seconds to stretch to
			new_length = float(arg)                       
			print 'new length is', new_length

	for sentence_file in range(start_file, stop_file+1):
		s = Sentence(str(sentence_file), new_length)       	# create a Sentence object s

		control_words = []

	   	for i in range(s.length):
	   		if s.new_length !=0:							# new_length of 0 means no stretch
		   		if i != s.length/2: 		 				# don't stretch the middle word
			   		new_l = s.new_length
			   		# new_l = base_l + gammavariate(k, theta)
			   		print 'new_l', new_l
			   		old_l = s.t_lengths[i]
			  		print 'old_l', old_l
			  		# stretch_f=old_l/new_l 				# use with stretch_simple
			   		stretch_f = new_l/old_l
			   		# s.words[i]=stretch_simple(s.words[i], stretch_f)
			   		s.words[i] = stretch(s.words[i], stretch_f)
			   		s.words[i] = fades(s.words[i], percentage_fade=0.1)
			   		control_words.append(s.words[i])
			   	else:
			   		old_l = s.t_lengths[i]
			   		stretch_f = base_l/old_l
			   		s.words[i] = stretch(s.words[i], stretch_f)
			   		s.words[i] = fades(s.words[i], percentage_fade=0.1)
			   		control_words.append(s.words[i])
			else:											# new_length of 0 means no stretch
				# stretch_f=1
		   		# s.words[i] = stretch(s.words[i], stretch_f)
		   		control_words.append(s.words[i])

		control_s = Control_sentence(control_words, s.freq, s.word_names) 					# feed data into class
	   	wav_sentence = s_concatenate(control_s, s.new_length, s.freq, str(sentence_file))	# use the class to create a .wav file of the sentence 

if __name__ == "__main__":
	main(sys.argv[1:])
