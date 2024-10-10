### Let's train a GPT on Marcus Aurelius's meditations!

This repo implements a simple decoder-transformer from "scratch" in Pytorch using Karpathy's NanoGPT as reference. Original text can be found [here](https://classics.mit.edu/Antoninus/meditations.mb.txt). I only removed the individual book titles and separators and cruft like that to get a nice clean dataset. 

The training logic along with hyperparameters can be found in `train.py`. To run simply execute `python train.py`. 



