# neural_filter
A Feed-forward Neural Network trained to learn a low-pass filter.

Here I tried to teach an FNN to learn a low-pass fiter for use on audio.

Initially I was going to mimic a typical filter but without an RNN, with 9 samples every input to the neural network input would have been shifted by one sample and the four samples either side of it taken. But this would have been tricky to feed into Keras without creating a really bloated dataset. I decided to first see if the network could learn a filter on chunks at a time, at a fixed frequency cutoff. I chose 600hz and 2,000hz.

If the network produced satisfactory results I would then maybe look at improving how the network samples the audio, e.g. as single amplitudes at a time with their neighbours rather than chunks at a time. Alas, I don't think the output is worth the effort.

While I don't provide examples, the output is like adding a noise filter which seemed to have some relevance to frequency response but only just enough to make it an "interesting" noise filter. That being said, noise is noise, and noise is not fun to listen to. I found that applying the same low-pass filter the original input data tried to teach the network to the output data helps a little but, it is what it is.

Generate your input files, I used [Audacity](https://www.audacityteam.org/), Generate > Noise, save that as `train_x.raw` as Unsigned 8-bit PCM, then apply a low-pass filter to it and save that as `train_y.raw` in the same format as last time. Then find a song, load it into Audacity, Tracks > Mix > Mix Stereo Down to Mono, then export that as `song.raw` as Unsigned 8-bit PCM again. Once these three file are placed in the same directory as `fit.py` you should now be able to execute `python3 fit.py 6 32 999 tanh adam 1 9 32`. The training process will also generate your first neural transformation of your `song.raw` and output it as `models/tanh_adam_6_32_9999_1_9_32.raw`.

## tips

- When loading new training data or similar be sure to delete any `.npy` files or the python scripts will load these pre-processed files rather than processing the new files. You can do this by running `clean.sh`.

- My personal opinion is that the best trained model is [gelu_adam_6_32_999_9_512_a41](models/9_sample/unsigned/gelu_adam_6_32_999_9_512_a41).

## how
You will need to put a few files in the working directory;
- `train_x.raw` - This should be the original audio file
- `train_y.raw` - This should be the original audio file but with a low-pass filter applied to it.
- `song.raw` - This should be some audio to apply the neural network to once it has trained.

## train
`python3 fit.py <layers> <units per layer> <batches> <activator> <optimiser> <cpu only> <samples> <epoches>`<br>
`python3 fit.py 6 32 999 tanh adam 1 9 32`

## generate
 `python3 gen.py <model path> <raw song path> <samples>`<br>
 `python3 gen.py models/9_sample/unsigned/gelu_adam_6_32_999_9_512_a41 song.raw`

