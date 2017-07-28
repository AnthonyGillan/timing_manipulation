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
import pywt
import numpy as np
from numpy.lib import stride_tricks
import peakutils
import matplotlib.pyplot as plt

class Sentence:

    def __init__(self, sentence_file, stretch_factor):
        text_file = open("sentences/"+sentence_file+".txt", "r") # open file, read mode
        lines = text_file.readlines()        # returns a list of lines in file, these files contain one word per line
        lines =[s.strip() for s in lines]    # trims trailing and leading whitespace
        text_file.close()        
        self.words = []                        # instance variable for words (audio)
        self.word_names = []                   # instance variable for words (text)
        word_freqs = []                        # instance variable for sample rates
        self.t_lengths = []                    # instance variable for time lengths of words
        self.samples_in_sentence = 0
        self.stretch_factor = stretch_factor
        print "\n"

        for word_name in lines:               # iterate through words (text) as only one word per line
            self.word_names.append(word_name) # append words (text) to list

            audio_seg_word = AudioSegment.from_file("words_audio/"+sentence_file+"/"+word_name+".wav") 	# open as audiosegment to strip silence 
            word_freq, w = wavfile.read("words_audio/"+sentence_file+"/"+word_name+".wav") 		   	   	# do this to get sample rate
            
            word = wave.open("words_audio/"+sentence_file+"/"+word_name+".wav")				# do this to convert to floating point 1 to -1
            astr = word.readframes(word.getnframes())
		    # convert binary chunks to short 
            word = struct.unpack("%ih" % (word.getnframes()* word.getnchannels()), astr)
            word = [float(val) / pow(2, 15) for val in word]
            word=np.asarray(word)

            start_trim_time = self.detect_leading_silence(audio_seg_word)						
            end_trim_time = self.detect_leading_silence(audio_seg_word.reverse())
            duration_time = len(audio_seg_word)

            # convert times to seconds (/1000) then to amount of samples
            start_trim = int(word_freq * (double(start_trim_time) / 1000))
            end_trim = int(word_freq * (double(end_trim_time) / 1000))
            duration = int(word_freq * (double(duration_time) / 1000))

            word = word[start_trim:duration - end_trim]				     # perform the trim
            
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

def s_concatenate(sentence):

    s = []

    for word in sentence.words:
        s = concatenate([s, word])                   # put words (audio) into array s

    s_int16 = zeros((s.shape[0]), dtype='int16')     # fill 0's (int16) in array with same number of rows as samples in all words here

    for i in range(0, s.shape[0]):                 # fill s with all int16 samples from 0 to #rows
        s_int16[i] = s[i]

    wavfile.write("new.wav",sentence.freq,s_int16) # (filename, sample rate, data array of int16 samples in sentence)
    
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

    result = ((2**(16-4)) * result / result.max()) # normalize (16bit)

    return result.astype('int16')

def stretch(word, stretch_factor):
    N = 2**10					# Number of channels
    M = 2**10					# Size of window
    w = np.hanning(M-1)			# Type of Window (Hanning)
    w = np.append(w, [0])		# Make window symmetric about (M-1)/2
    Os = 8.						# Synthesis hop factor 
    Rs = int(N / Os)			# Synthesis hop size
    alpha = stretch_factor

    pv = PhaseVocoder(N, M, Rs, w, alpha)
    new_w = pv.timestretch(word, alpha)  		  # stretch word by factor alpha

    new_w = ((2**(16-4)) * new_w / new_w.max()) # normalize (16bit)

    return new_w.astype('int16')

    return new_w

def peak_envelope(sound, peak_separation, min_amplitude):
	# envelope=np.zeros(sound.shape)
	# indexes=[]
	# peaks=[]
	# indexes=peakutils.indexes(sound, thres=10, min_dist=20)
	# for k in range(0, len(indexes)):
	#  	peaks.append(sound[indexes[k]])

	# interpolation_function = interp1d(indexes,peaks, kind='cubic', bounds_error=False, fill_value=0.0)

	# # Evaluate each model over the domain of (s)
	# for k in xrange(0,len(sound)):
	#     envelope[k]=interpolation_function(k)

	# return envelope

	filtered_wav=sound
	envelope=np.zeros(filtered_wav.shape)

	# Prepend the first value of (s) to the interpolating values. 
	# This forces the model to use the same starting point for both the upper and lower envelope models.
	# Lists with the same starting point

	peak_x = [0,]
	peak_y = [filtered_wav[0],]
	spaced_peak_x = []
	spaced_peak_y = []

	# Detect peaks and mark their location in peak_x, peak_y
	# xrange better for large range 

	for k in xrange(0,len(filtered_wav) - 1):
	    if (sign(long(filtered_wav[k]) - long(filtered_wav[k-1])) == 1 
	    	and sign(long(filtered_wav[k]) - long(filtered_wav[k+1])) == 1):	# get peak
			peak_x.append(k)
			peak_y.append(filtered_wav[k])
		        if (k>2 and (peak_x[-1] - peak_x[-2]) > peak_separation):		# if peaks arre separated by
		        	if (filtered_wav[k] > min_amplitude):						# if the peaks are above a min
			        	spaced_peak_x.append(k)
			        	spaced_peak_y.append(filtered_wav[k])

	# Append the last value of (wav_sentece) to the interpolating values. 
	# This forces the model to use the same ending point for the envelope model.
	# [-1] is the last element of a list

	peak_x.append(len(filtered_wav) - 1) 	# last value of index of wav_sentence
	peak_y.append(filtered_wav[-1])		# last value of wav_sentence

	# Fit suitable model to the data.
	# interp1d creates a 1D interpolation function to find value of new points

	interpolation_function = interp1d(spaced_peak_x,spaced_peak_y, kind='slinear', bounds_error=False, fill_value=0.0)

	# Evaluate each model over the domain of filtered_wav
	for k in xrange(0, len(filtered_wav)):
	    envelope[k] = interpolation_function(k)

	rms_envelope = window_rms(envelope, window_size=100)
	rms_envelope = window_rms(rms_envelope, window_size=200)
	return rms_envelope

def window_rms(signal, window_size):
	a2 = np.power(signal,2)
	window = np.ones(window_size)/float(window_size)
	return np.sqrt(np.convolve(a2, window, 'valid'))

def fft(signal, samp_freq, N):
	f = np.fft.rfft(a=signal, n=N, axis=0)
	f = f/max(abs(f)) 						# scale amplitude
	power_density = abs(f)**2
	return power_density

def butter_lowpass(signal, lowcut, samp_freq, order):
	nyq = 0.5 * samp_freq                      
	low = lowcut / nyq   
	b, a = butter(N=order, Wn=low, btype='lowpass') 
	filtered_signal = lfilter(b, a, signal)
	return filtered_signal

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order) # obtain filter transfer function
    y = lfilter(b, a, data)                                  # butter_bandpass_filter, data is sample array
    return y                                                 # return filtered signal

def main(argv):

	base_l = 0.15                                           # 150ms base length
	new_l_av = 0.2                                          # new average length 
	k = 20                                                  # shape of distribution
	theta = new_l_av/k                                      # scale of distribution

	sentence_file = ''
	stretch_factor = 1.0

	num_gammatone_bands = 32
	gammatone_cutoff_l = 10

	try:                                                   # runs in entirety unless exception
		opts, args = getopt.getopt(argv,"hi:s:",["sfile=","stretch="])   # Parses command line opts and param list. (args, shortopts, longopts=[]), short -h,-i: colon if requires argument
	except getopt.GetoptError:                             # raised when an unrecognized option is found in the argument list or when an option requiring an argument is given none
		print 'load_sentence.py -i <sentence_file> -s <stretch_factor(float)>'         # <input file>
		sys.exit(2)                                         # exit from python -> arg is an int (2) giving the exit status
	for opt, arg in opts:                                  # cycle through options
		if opt == '-h':
			print 'load_sentence.py -i <sentence_file -s <stretch_factor(float)>'     # <input file>
			sys.exit()                                      # exit from python
		elif opt in ("-i", "--sfile"):                      # if option is -i ie// input
			sentence_file = arg                              # sentence file = arguments
			print 'Sentence file is', sentence_file           
		elif opt in ("-s", "--stretch"):                      # if option is -i ie// input
			stretch_factor = float(arg)                       # stretch facotr = arguments
			print 'Stretch factor is', stretch_factor

	s = Sentence(sentence_file, stretch_factor)                              # create a Sentence object s

	# s_concatenate(s)

	# control_words = [0]*s.samples_in_sentence
	control_words = []

   	for i in range(s.length):
   		new_l=s.stretch_factor #base_l+gammavariate(k, theta)
   		print 'new_l', new_l
   		old_l=s.t_lengths[i]
  		print 'old_l', old_l
#####################################################################################################################################################
  		# stretch_factor=old_l/new_l # eg stretch factor 0.5 -> 0.5 times original
   		stretch_factor = new_l/old_l
   		# s.words[i]=stretch_simple(s.words[i], stretch_factor)
   		s.words[i] = stretch(s.words[i], stretch_factor)
   		control_words.append(s.words[i])

	control_s = Control_sentence(control_words, s.freq, s.word_names) 	# feed data into class
   	wav_sentence = s_concatenate(control_s)								# use the class to create a .wav file of the sentence 

   	samples_in_manip_sentence = len(wav_sentence)

	a = plt.subplot(411)								# plot the sonogram
	dt = 1.0/(s.freq)
	t_end = (float(samples_in_manip_sentence)/s.freq)
	t = np.arange(0.0, t_end, dt)
	a.set_title('Waveform of Syllable Stretched to 500ms - Should Show 2 Hz Peak')
	a.set_xlabel('t [s]')
	a.set_ylabel('amplitude')
	a.grid(b=None, which='both', axis='both')
	plt.plot(t, wav_sentence, color='red')

	c = plt.subplot(412)								# plot the spectrum taken over the whole sample
	fft_sentence = fft(signal=wav_sentence, samp_freq=s.freq, N=44100)
	c.set_xscale('log')
	c.set_xlabel('frequency [Hz]')
	c.set_ylabel('|amplitude|')
	plt.plot(fft_sentence)
				
	d = plt.subplot(413)					# plot the zeroed envelope of the whole sample
	# envelope=peak_envelope(sound=wav_sentence, peak_separation=5, min_amplitude=10)
	# hilbert envelope
	# analytic_signal = hilbert(wav_sentence)        	
	# envelope = np.abs(analytic_signal)

	# decomposed, reconstructed envelope using gammatone
	# create a gammatone filterbank
	coefs = gammatone_create(samp_freq=s.freq, num_freqs=num_gammatone_bands, lower_cutoff=gammatone_cutoff_l)
	envelope = control(s.freq, wav_sentence, new_length=1, gammatone_coeffs=coefs, num_gammatone_bands=num_gammatone_bands)

	# smooth
	envelope = window_rms(signal=envelope, window_size=200) # smooth the envelope

	this_max = np.amax(abs(envelope))
	envelope = envelope/this_max
	zeroed_envelope = envelope-np.mean(envelope)
	dt = 1.0/(s.freq)
	t_end = (float(len(zeroed_envelope))/s.freq)
	t = np.arange(0.0, t_end, dt)
	d.set_title('Normalised, Smoothed, Zeroed Syllable Waveform Envelope')
	d.set_xlabel('t [s]')
	d.set_ylabel('amplitude')
	d.grid(b=None, which='both', axis='both')
	plt.plot(t, zeroed_envelope, color='blue')

	e = plt.subplot(414)					# plot the envelope's spectrum			                            
	fft_envelope = fft(signal=zeroed_envelope, samp_freq=s.freq, N=44100)
	e.set_xscale('log')
	e.set_title('FFT spectrum of Zeroed Syllable Waveform Envelope')
	e.set_xlabel('frequency [Hz]')
	e.set_ylabel('|amplitude|')
	e.grid(b=None, which='both', axis='both')
	plt.plot(fft_envelope, color='green')

	plt.tight_layout() 					# space the plots so they don't overlap


	fig2 = plt.figure()					# plot the lowpass filtered sample's spectrum on new figure								
	g = fig2.add_subplot(111)			    # do this to look more closely at magnitude of lower frequencite
	filtered_wav_sentence = butter_lowpass(signal=wav_sentence, lowcut=300, samp_freq=s.freq, order=10)
	fft_filtered_sample = fft(signal=wav_sentence, samp_freq=s.freq, N=44100)
	g.set_xscale('log')
	g.set_title('FFT Frequency Spectrum of Stretched (1.9x Original) Syllable')
	g.set_xlabel('frequency [Hz]')
	g.set_ylabel('|amplitude|')
	g.grid(b=None, which='both', axis='both')
	g.plot(fft_filtered_sample, color='orange')

	fig5 = plt.figure()					# plot power spectrogram and obtain spectrogram
	i = fig5.add_subplot(111)
	### Parameters ###
	nyq = s.freq/2
	fft_size = 2**10 # window size for the FFT
	step_size = fft_size/16 # distance to slide along the window (in time)
	spec_thresh = 4.3 # threshold for spectrograms in dB. increase values below thresh to thresh. lower=less noise.

	power_specgram, specgram, num_samples, num_windows = pretty_spectrogram(wav_sentence.astype('float64'), fft_size=fft_size, 
                                   step_size=step_size, log=True, thresh=spec_thresh)
	oriented_specgram = np.transpose(specgram)
	oriented_power_specgram = np.transpose(specgram)
	time_len_fft = float(num_samples)/s.freq
	# create image of spectrogram matrix with correct axes
	axes_image = i.matshow(oriented_power_specgram, interpolation='gaussian', aspect='auto', 
		cmap=plt.cm.nipy_spectral, origin='lower', extent=(0, time_len_fft, 0, nyq))
	
	fig5.colorbar(axes_image)
	i.set_title('Original Spectrogram')
	i.set_ylabel('Frequency [Hz]')
	i.set_xlabel('Time [sec]')

	fig6 = plt.figure()			# plot 2d FFT of spectrogram
	j = fig6.add_subplot(111)
	# time sampling freq of spectrogram is num_fft_windows/signal_length_time (Hz)
	samp_freq_time = float(num_windows)/time_len_fft
	samp_freq_time_nyq = samp_freq_time/2
	# time sampling freq of spectrogram is num_fft_windows/signal_length_time (cycles/kHz)
	samp_freq_freq = (float((fft_size/2))/nyq)*1000
	samp_freq_freq_nyq = samp_freq_freq/2
	# fft2
	fft_2d = np.fft.fft2(oriented_power_specgram)
	# shift so lof freqs are in center
	# USE FOR FILTERING AND RECOVERY
	fft_2d = np.fft.fftshift(fft_2d)
	# log transform to scale down massive low freq values for display
	# DISPLAY VALUES
	mod_specgram_disp = np.log10(np.abs(fft_2d))
	# plot modulation amplitude 
	axes_image = j.matshow(mod_specgram_disp, interpolation='gaussian', aspect='auto', 
		cmap=plt.cm.nipy_spectral, origin='lower', 
		extent=(-samp_freq_time_nyq, samp_freq_time_nyq, -samp_freq_freq_nyq, samp_freq_freq_nyq))

	j.set_title('Shifted log10|2D FFT| of Spectrogram')
	j.set_ylabel('Spectral Modulation [cyc/kHz]')
	j.set_xlabel('Temporal Modulation [Hz]')

	fig7 = plt.figure()			# plot reconstructed, filtered spectrogram
	k = fig7.add_subplot(111)
	# create vertical filter mask and multiply with signal in Fourier domain
#####################################################################################################################################################
	filt = butter2d_vert_lp(shape=fft_2d.shape, f=100, n=10, pxd=1)
	fft_filt_sig = fft_2d * filt
	# shift back
	recon_specgram = np.fft.ifftshift(fft_filt_sig)
	# ifft2
	recon_specgram = np.fft.ifft2(recon_specgram)
	# abs and invert (-)
	recon_specgram = -np.abs(recon_specgram)
	recon_specgram[recon_specgram < -4.1] = -4.2

	axes_image = k.matshow(recon_specgram, interpolation='gaussian', aspect='auto', 
		cmap=plt.cm.nipy_spectral, origin='lower', extent=(0, time_len_fft, 0, nyq))

	fig7.colorbar(axes_image)
	k.set_title('Recovered Spectrogram')
	k.set_ylabel('Frequency [Hz]')
	k.set_xlabel('Time [sec]')

	recovered_audio_orig = invert_pretty_spectrogram(np.transpose(recon_specgram), 
		fft_size=fft_size, step_size=step_size, log=True, n_iter=10)

	recovered_audio_orig = louden(recovered_audio_orig)

	# recovered_audio_orig = butter_lowpass(recovered_audio_orig, 8000, s.freq, 10)

	fig8 = plt.figure()			# plot reconstructed audio waveform
	l = fig8.add_subplot(111)
	# dt=1.0/(s.freq)
	# t_end=(float(samples_in_manip_sentence)/s.freq)
	# t=np.arange(0.0, t_end, dt)
	# l.set_title('Waveform of Syllable Stretched to 500ms - Should Show 2 Hz Peak')
	# l.set_xlabel('t [s]')
	# l.set_ylabel('amplitude')
	# l.grid(b=None, which='both', axis='both')
	l.plot(recovered_audio_orig, color='red')

	wavfile.write("recov.wav",s.freq, recovered_audio_orig) # (filename, sample rate, data array of int16 samples in sentence)


	plt.show()	# show all the plots

	# stretch factor set to 1 hardcode
	# oriented_power_specgram=oriented_specgram for invertibility

if __name__ == "__main__":
	main(sys.argv[1:])
