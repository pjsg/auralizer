This is a program (the auralizer) that auralizes the CSV files generated by Grape 2 radios into a 16-bit WAV audio file (the auralization). 

Each of the grape radios is mapped to its own frequency in the audio band, and then the frequency deviation is 
magnified so that the small (single Hz) changes in received frequency are clearly audible. The volume
of the output tone is directly scaled from the Vrms reported by the Grape 2 radio. Each of the three channels
is volume normalized so that they all appear to have the same volume.

This is very much version 0.1 -- and I did this on the eclipse day before driving off to see the eclipse and a little bit more after we got back.

Thanks to Dana K8YUM for the word `auralization` which led to the obvious word `auralizer`.

A standard invocation might be:

```
python auralizer.py 2024-04-08T000000Z_N0001011_G2R*.csv --out auralizer-2024-04-08.wav
```

Philip Gladstone, N1DQ
