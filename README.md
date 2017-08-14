# timing_manipulation
Software for creating syllable-timing-manipulated sentences for a psychophysical study on speech comprehensibility.

# Creating timing-manipulated sentences.
Use create_sentence.py to put together sentences as specified in sentences longer (or shorter if the directory is changed).

# Command line arguments for create_sentence.py:

-b
start sentence num  -> first sentence to assemble (1 is the first sentence not 0)
-e
end sentence num    -> last sentence to assemble
-n
new word length (s) -> length in senconds for non-target words to be stretched to
-t
target word number -> location of the target word in the sentence eg// 1 for word 1

The target word is the word which will be stretched to a pre-specified base length (base_l)

create_sentence.py -b <start sentence num> -e <end sentence num> -n <new word length (s) for words other than target> -t <target word number>


