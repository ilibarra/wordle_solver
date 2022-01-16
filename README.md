## Comparison of strategies for Wordle

[Wordle app](https://www.powerlanguage.co.uk/wordle/)

The goal of this repo is to compare, in terms of performance at getting reliable solutions to wordle.

<a href="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" target="_blank"><img src="https://github.com/ilibarra/wordle_solver/blob/main/data/clustering_example.png" alt="Clustering example" id="bg" width="250px" height="325px" /></a>


Example of strategies:
a. Letter frequency, position independent.
b. Letter frequency per position, position-specific.
c. Letter frequency per position, plus letter co-variation between positions.

Below, there is a simulation to show that (c) > (b) > (a), on average.
This is dependent on the current dictionary
- [Wordle dictionary ~2,000 words](https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts),
- [Linux American English ~6,000 words](data/american-english).

Changes in the number of letters per word (5) and dictionary complexity should generate slight variations in results.

**Troubleshooting**: Please open an [issue](https://github.com/ilbarra/wordle_solver/issues).
**License**: [MIT](https://github.com/ilibarra/wordle_solver/blob/main/LICENSE).