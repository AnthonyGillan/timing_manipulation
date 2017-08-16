# timing_manipulation
Software for creating and analysing syllable-timing-manipulated sentences for a psychophysical study on speech comprehensibility.

## words_audio_...
After 33 sentences were composed (**results_and_sentence_material/sentence_material_longer.txt**), all of the words within these sentences were order-randomised and formatted into columns for ease of reading (**results_and_sentence_material/words_for_reading_slides.pptx**) for the recording session. <br />

**words_for_reading_slides.pptx** was read in its entirety (with some small sections re-read to correct errors) three times and recorded with a Rode NT1A microphone to Logic Pro X through a MOTU Ultralite MKIII audio interface. <br />

The recordings were edited and the clearest and most neutrally read version of each word was selected, lightly volume faded at beginning and end, de-noised with iZotope RX Noise and exported as stereo interleaved 16 bit, 44.1 kHz wave files (**words_audio_stereo_denoised_jess**). <br />

Audacity was used to create a batch processing chain, passing the files through iZotope Nectar for dynamic range parallel compression, de-essing, EQ-ing, light limiting and finally conversion from stereo to mono files for use with the python programs. These processed files are in the **words_audio_jess** folder. <br />

**words_audio_stereo_polly** contains words synthesised by AWS polly for the first iteration of the software. **words_audio_polly** contains mono versions.

## Creating timing-manipulated sentences with create_sentence.py
Use **create_sentence.py** to put together sentences as specified in .txt files in the **sentences_longer** folder (or **sentences_shorter** if the directory is changed and the audio files directory is changed to **words_audio_polly**). The .txt files specify the content of the sentence and the corresponding word audio files from words_audio_jess are concatenated.

## Command line arguments for create_sentence.py
**create_sentence.py** **-b** start sentence num **-e** end sentence num **-n** new word length **-t** target word number <br />

**-b** <br />
start input sentence num -> first sentence to assemble (1 is the first sentence not 0) <br />
**-e** <br />
end input sentence num   -> last sentence to assemble <br />
**-n** <br />
new word length          -> length in seconds for non-target words to be stretched to <br />
**-t** <br />
target word number       -> location of the target word in the sentence eg// 1 for word 1 <br />

The target word is the word which will be stretched to a pre-specified base length (base_l) <br /

## Creating timing-manipulated sentences with create_sentence_a.py
This program differs from **create_sentence_analysis.py** in that stretching of words is done with respect to the average word length (pre-defined). A deviation factor is used (**-d** in the command line arguments). When **deviation_factor = 0**, all words are stretched to the average word length **ave_l**. The higher the deviation factor, the more the words deviate in length from **ave_l**. 

## Command line arguments for create_sentence_a.py
**create_sentence_a.py** **-b** start sentence num **-e** end sentence num **-d** deviation factor **-t** target word number <br />

## Spatial filtering in create_sentence.py and create_sentence_a.py
**butter2d_horiz_lp**    -> horizontal spatial filter <br />
**butter2d_vert_lp**     -> vertical spatial filter <br />

Filter parameters are described in **spatial_filters.py** <br />

audio files created from spatially filtered signals are named -> 'sentence_num_recov.wav'


## Analysis and manipulation with create_sentence_analysis.py
This program acts on one sentence. It may be used to perform various spectro-temporal analyses on a sentence put together as specified in .txt files in the **sentences_longer folder** (or **sentences_shorter** if the directory is changed and the audio files directory is changed to **words_audio_polly**).

## Analysis and manipulation with create_sentence_analysis_a.py

## Command line arguments for create_sentence_analysis_a.py
**create_sentence_analysis.py** **-i** input sentence num **-d** deviation factor

## Spatial filtering in create_sentence_analysis.py and create_sentence_analysis_a.py
2D FFT operations and spatial filtering operations are implemented between this line:<br />
############################ plot 2d FFT of spectrogram ############################<br />

and this line:<br />
############################ create .wav of spectro-temporal-modulation manipulated signal ############################<br />

**butter2d_horiz_lp**    -> horizontal spatial filter <br />
**butter2d_vert_lp**     -> vertical spatial filter <br />

Filter parameters are described in **spatial_filters.py**

## Command line arguments for create_sentence_analysis.py
**create_sentence_analysis.py** **-i** input sentence num **-n** new word length <br />

**-i** <br />
input sentence number    -> the sentence to be manipulated and analysed <br />
**-n** <br />
new word length          -> length in seconds for non-target words to be stretched to <br />

## Phase vocoding with pv.py
A slightly modified version of: https://github.com/multivac61/pv to allow for time compression to much shorter values. <br />

Phase-locked phase vocoding: http://msp.ucsd.edu/Publications/mohonk95.pdf <br />

## Visualising the spatial filters in spatial_filters_plots.py
run **spatial_filters_plots.py** to visualise spatial filters.
