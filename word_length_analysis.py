import sys, getopt
from pydub import AudioSegment
from pylab import *
from scipy.io import wavfile
import numpy as np

class Sentence:

    def __init__(self, sentence_file):
        text_file = open("sentences/"+sentence_file+".txt", "r") # open file, read mode
        lines = text_file.readlines()        # returns a list of lines in file, these files contain one word per line
        lines =[s.strip() for s in lines]    # trims trailing and leading whitespace
        text_file.close()        
        self.words = []                        # instance variable for words (audio)
        self.word_names = []                   # instance variable for words (text)
        word_freqs = []                        # instance variable for sample rates
        self.t_lengths = []                    # instance variable for time lengths of words
        self.samples_in_sentence = 0
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
            self.t_lengths.append(double(word.shape[0]) / word_freq)     # (number of rows in sample data ie samples)/(sample rate) = time length of word
            self.samples_in_sentence += word.shape[0]
            print word.shape,word_freq,self.t_lengths[-1],word_name      # number of samples, sample rate, time legth[last in list], word (text)

        self.freq = word_freqs[0]                                        # first sample rate of word
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

def main(argv):
	sentence_file = ''

	try:                                                   # runs in entirety unless exception
		opts, args = getopt.getopt(argv,"hi:",["sfile="])   # Parses command line opts and param list. (args, shortopts, longopts=[]), short -h,-i: colon if requires argument
	except getopt.GetoptError:                             # raised when an unrecognized option is found in the argument list or when an option requiring an argument is given none
		print 'load_sentence.py -i <sentence_file>'         # <input file>
		sys.exit(2)                                         # exit from python -> arg is an int (2) giving the exit status
	for opt, arg in opts:                                   # cycle through options
		if opt == '-h':
			print 'load_sentence.py -i <sentence_file>'     # <input file>
			sys.exit()                                      # exit from python
		elif opt in ("-i", "--sfile"):                      # if option is -i ie// input
			sentence_file = arg                              # sentence file = arguments
			print 'Sentence file is', sentence_file           

	s = Sentence(sentence_file)                              # create a Sentence object s
 	words_and_lengths = zip(s.t_lengths, s.word_names)
 	words_and_lengths.sort()
 	words_and_lengths_no_dup = []

	for i in words_and_lengths:
		if i not in words_and_lengths_no_dup:
			words_and_lengths_no_dup.append(i)

	f = open("word_lengths.txt","w+")
	for i in range(len(words_and_lengths_no_dup)):
		f.write("%s\t%s\r\n" % (words_and_lengths_no_dup[i][1], words_and_lengths_no_dup[i][0]))
	f.close()

if __name__ == '__main__':
	main(sys.argv[1:])
