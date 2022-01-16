## Comparison of strategies for Wordle

[Wordle app](https://www.powerlanguage.co.uk/wordle/)

The goal of this repo is to compare, in terms of performance, strategies that minimize the number of guesses needed to find a word in Wordle. Strategies tested assume that we know the dictionary of words.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" alt="Clustering example" id="bg" width="250px" height="325px" /></a>


### Example of strategies:

a. Letter frequency, position independent.
b. Letter frequency per position, position-specific.
c. Letter frequency per position, plus letter co-variation between positions.

Below, there is a simulation to show that (c) > (b) > (a), on average.
This is dependent on the current dictionary
- [Wordle dictionary ~2,000 words](https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts),
- [Linux American English ~6,000 words](data/american-english).

#### Usage

First run the daily.py script without any guesses. You will get the most likely guess, given the input strategy
```
python daily.py -d american_5 --strategy posfreqcovar
```

Assuming a guess BRINY query that into Wordle. Then you will get rules based on matches back, that you can input to the script
```
python daily.py -g "BRINY" --rules "01000" -d american_5 --strategy posfreqcovar
```

Assuming the next word is SERUM, one match in position one
```
python daily.py -g "BRINY,SERUM" --rules "01000,20100" -d american_5 --strategy posfreqcovar
```

Continue until getting solution


### Simulation results

So far, strategies tested show a slight advantage towards using letter positional frequency as the main criteria for selection. Even though I have tested this with positional dependencies being weighted, under the asssumption that this would give a slight improvement, at the end an overall selection based on the most frequent letter per position is not conferring an advantage as a strategy. This could change in case there is a bug on the code, or the best strategy changes based on the length of words and letter co-variation among words.

**Troubleshooting**: Please open an [issue](https://github.com/ilbarra/wordle_solver/issues).
**License**: [MIT](https://github.com/ilibarra/wordle_solver/blob/main/LICENSE).