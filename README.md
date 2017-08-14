# timing_manipulation
Software for creating syllable-timing-manipulated sentences for a psychophysical study on speech comprehensibility.

# Creating timing-manipulated sentences.
Use create_sentence.py to put together sentences as specified in sentences longer (or shorter if the directory is changed).

# Command line arguments for create_sentence.py:

-b <br />
start sentence num  -> first sentence to assemble (1 is the first sentence not 0) <br />
-e <br />
end sentence num    -> last sentence to assemble <br />
-n <br />
new word length     -> length in seconds for non-target words to be stretched to <br />
-t <br />
target word number  -> location of the target word in the sentence eg// 1 for word 1 <br />

The target word is the word which will be stretched to a pre-specified base length (base_l) <br />

create_sentence.py -b <start sentence num> -e <end sentence num> -n <new word length (s) for words other than target> -t <target word number>

# Spatial filtering in create_sentence.py:

butter2d_horiz_lp -> horizontal spatial filter <br />
butter2d_vert_lp  -> vertical spatial filter <br />

audio files created from spatially filtered signals are named -> 'sentence_num_recov.wav'
