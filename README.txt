THIS IS FOR FACE RECOGNITION USING OPENCV 2.4.3

ONLY ALGORITHM CHANGES WILL BE DISPLAYED HERE!
----------------------------------------------------------------
2/14/2013

- Give up using input_align.txt etc.. to control input training and matching pics.
- Totally rearranged VC solution to build match and training programs at the same time.

2/18/2013

- Add uniform LBP patterns to make histogram statistically efficient( performance degraded? ) // disabled

2/19/2013

- Add threshold to LBP comparison. (no big change)

2/20/2013

- Add distance normalization to histogram comparison [ d = (h1-h2)^2 / (h1+h2) ] //disabled
- Add GBP features( global binary patterns, which is the binary comparison with global mean) // disabled

2/27/2013

- Add Gabor Feature extraction module

3/3/2013

- Add Flip matching mode
- Add camera test module

3/5/2013

- Add rotation invariant feature extraction
- Add LBP window size controll

3/6/2013

- Add Historgram Equalization

3/12/2013

- Add Local Gabor Texton vocabularies training module