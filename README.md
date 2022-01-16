## Wordle Solver - Strategies 

### Motivation
The goal of this repository is to compare, in terms of performance, strategies that minimize the number of guesses needed to find a word match in Wordle. A script for general usage on is also available.

### Introduction
While playing [Wordle's word of the day](https://www.powerlanguage.co.uk/wordle/), several strategies and dictionaries can be used to guide the selection of a best next guesses. To describe strategies, in the example below, the target word is JEWE. In the current iteration and after an initial guess, we ended up with 10 possible words.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" alt="Clustering example" id="bg" width="250px" height="325px" /></a>

Among possible strategies, one could try:

a. Letter frequency, position-independent.
b. Letter frequency per position, position-specific.
c. Letter frequency per position, plus letter co-variation between positions.
d. Discarding words submitted at previous days (only via scripts)

The reason I implemented this was to test option *c. letter co-variation among positions*. Visually, this can be described by checking the low overlap of E at positions 2 and 3, which is an indicator that those two letters are so co-occurring together, and might be used simultaneously to guide the selection of the best next guess. For more details, see [mutual information](https://en.wikipedia.org/wiki/Mutual_information).

Below, there is a simulation to test both a, b, and c, in the next dictionaries. I am not sure which is the official dictionary that Wordle uses, but as the Linux one has more words I am using that one by default.
- [Linux American English ~6,000 words](data/american-english).
- [Wordle dictionary ~2,000 words](https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts),

### Simulation results

So far, strategies that use letter positional frequency as main criteria for selection are having the lower number of guesses. Letter co-variation among positions, so far, is not conferring an positive, and it reduces the overall performance in terms of guesses. This trend could change in case there is a bug in the code, or a better strategy is based on the words complexity and co-variation in the dictionary.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/data/benchmark_results.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/data/benchmark_results.png" alt=“Benchmarking” id=“bg” width=“600px” height=“300px” /></a>

#### Usage
First, run the daily.py script without any input guesses. You will get the most likely guess, given the input strategy and dictionary.
```
python daily.py -d american_5 --strategy posfreqcovar
```

Assuming a guess BRINY query that into Wordle. Then you will get rules based on matches back, that you can input into the script
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
**License**: [MIT](https://github.com/ilibarra/wordle_solver/blob/main/LICENSE).

