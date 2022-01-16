## Comparison of strategies for Wordle

[Wordle app](https://www.powerlanguage.co.uk/wordle/)

The goal of this repo is to compare, in terms of performance at getting reliable solutions to wordle.

<a href="https://github.com/ilibarra/wordle_solver/data/clustering_example.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/data/clustering_example.png" alt="Clustering example" id="bg" width="280px" height="45px" /></a>


Example of solutions:

- (i) Overall letter frequency.
- (ii) Letter frequency per position
- (iii) Letter frequency per position + mutual information per positions.

Below, there is a simulation to show that (iii) > (ii) > (i), average. This is dependent on the current dictionary ([Wordle dictionary ~2,000 words](https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts), and [Linux American English](data/american-english ~6,000 words)) and number of letters per word (5). Slight variations in results and suggestions could happen if changing either of those two.