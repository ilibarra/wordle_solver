## Solver and comparison of strategies for Wordle

### Motivation
The goal of this repository is to compare, in terms of performance, strategies that minimize the number of guesses needed to find a word match in Wordle. A script for general usage on is also available.

### Introduction
While playing [Wordle's word of the day](https://www.powerlanguage.co.uk/wordle/), several strategies and dictionaries can be used to guide the selection of a best next guesses. To describe strategies, in the example below, the target word is JEWE. In the current iteration and after an initial guess, we ended up with 10 possible words.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" alt="Clustering example" id="bg" width="330px" height="550px" /></a>

Among possible strategies, one could try:

1. Letter frequency, position-independent.
2. Letter frequency per position, position-specific.
3. Letter frequency per position, plus letter co-variation between positions.
4. Brute force mapping of best word matched similar to [Tyler Gaiel's implementation](https://medium.com/@tglaiel/the-mathematically-optimal-first-guess-in-wordle-cbcb03c19b0a) (*pending*).
5. Discarding words submitted at previous days (only via scripts)

I implemented this to test option *3. letter co-variation among positions*. Visually, this can be described by checking the low overlap of E at positions c.3 and c.4. This is an indicator that those two letters are not co-occurring together, and might be used simultaneously to guide the selection of the best next guess. For more details, see [mutual information](https://en.wikipedia.org/wiki/Mutual_information).

Below, there is a simulation to test both 1, 2, and 3, with two public dictionaries. I am not sure which is the official dictionary that Wordle uses, but as the Linux one has more words I am using that one by default.
- [Linux American English ~6,000 words](data/american-english).
- [Wordle dictionary ~2,000 words](https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts),

### Results

Tests using all words from the dictionaries indicate that overall letter frequencies (*wordfreq*) are the most relevant criteria for best next guess selection (i.e. lowest median guesses). Letter co-variation among positions, so far, is not conferring a positive advantage, and it seems to perform worse overall (lowest mean, ~3.68). This trend could change in case there is a bug in the code, or a better strategy changes based on the co-variation complexity of words in the dictionary.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/out/benchmarking_5letters.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/out/benchmarking_5letters.png" alt=“Benchmarking” id=“bg” width=“200px” height=“600px” /></a>

*(blue line = median, red line = mean)*

This is the same analysis, across dictionaries of length 3, 4 and 5. Overall, trends do not indicate that co-variation improves results. The strategy *wordfreq* seems to repeatedly word well.


<a href="https://github.com/ilibarra/wordle_solver/blob/main/out/benchmarking_results.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/out/benchmarking_results.png" alt=“Benchmarking” id=“bg” width=“600px” height=“300px” /></a>

#### Next steps

- Addition of best guesses based on brute force.

#### Usage
First, run the daily.py script without any input guesses. You will get the most likely guess, given the input strategy and dictionary. Heatmaps saving clustering of remainings words will be saved in `out`.
```
python daily.py -g '' -r '' -d american_5 --strategy posfreqcovar
```

Assuming as guess the word BRINY, then query that into Wordle. You will get rules based on matches to the word of the day, that you can use as input in the script (0 = no match, 1 = word match, 2 = position match). Additional, heatmaps with the visualization above will be saved in out, so you can visualize the current options.
```
python daily.py -g "BRINY" --rules "01000" -d american_5 --strategy posfreqcovar
```

Assuming the next word is SERUM, one match in position one
```
python daily.py -g "BRINY,SERUM" --rules "01000,20100" -d american_5 --strategy posfreqcovar
```
From here, you can continue until getting a solution (probably 1-2 more guesses, max.)

If you think there could be additional strategies to test, reach out! Have fun!

**Troubleshooting**: Please open an [issue](https://github.com/ilbarra/wordle_solver/issues).
**License**: [GNU](https://github.com/ilibarra/wordle_solver/blob/main/LICENSE).

