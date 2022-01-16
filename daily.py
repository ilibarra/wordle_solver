import argparse
import wordle

if __name__=='__main__':
    """
    Get next guesses assuming dictionary, strategy and previous output
    """
    
    parser = argparse.ArgumentParser(description='Infer next best guess')

    parser.add_argument('-g', '--guesses', nargs='+', required=True, help='a comma separate set of words tested')
    parser.add_argument('-r', '--rules', required=False, help='Rules for the rest of words, in format 0 (gray, no match), 1 (yellow, word match) and 2 (green, position match) e.g. 00000,01002')
    parser.add_argument('-s', '--strategy', required=False, default='posfreqcovar',
                        help='strategy to use [wordfreq, posfreq posfreqcovar]')
    parser.add_argument('-d', '--dictionary', required=False, default='wordle_5',
                        help='word dictionary')

    args = parser.parse_args()
    
    guesses = args.guesses[0].split(',')
    rules = args.rules.split(',')
    
    assert len(guesses) == len(rules)
    
    words_by_dictionary, _ = wordle.load_dictionaries()
    remaining_words = words_by_dictionary[args.dictionary]
    
    print('# words in dictionary:', len(remaining_words))
    
    rules_encoded = []
    for i, w in enumerate(guesses):
        print('\nGUESS %i: %s' % (i, w))
        curr_rules = wordle.get_rules(w, rules[i])
        # print(curr_rules)
        remaining_words = wordle.select_multiple_rules(remaining_words, curr_rules)
        rules_encoded += curr_rules
        print('# remaining words in dictionary:', len(remaining_words))
        # print(remaining_words)
    
    
    basename = str('-'.join([args.dictionary, "_".join(guesses), '_'.join(rules)]))
    out_basename = 'out/' + basename
    print('Heatmaps saved at %s' % basename)
    wordle.plot_words(remaining_words, out_basename=out_basename, plot=True)
    
    guess = wordle.infer(remaining_words, args.strategy, sort_ascending=False)
    print('best guess (with strategy %s): %s' % (args.strategy, guess))