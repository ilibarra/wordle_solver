import argparse
import wordle

if __name__=='__main__':
    """
    Get next guesses assuming dictionary, strategy and previous output
    """
    
    parser = argparse.ArgumentParser(description='Infer next best guess')

    parser.add_argument('-g', '--guesses', nargs='+', required=False, help='a comma separate set of words tested')
    parser.add_argument('-r', '--rules', required=False, help='Rules for the rest of words, in format 0 (gray, no match), 1 (yellow, word match) and 2 (green, position match) e.g. 00000,01002')
    parser.add_argument('-s', '--strategy', required=False, default='posfreqcovar',
                        help='strategy to use [wordfreq, posfreq posfreqcovar]')
    parser.add_argument('-d', '--dictionary', required=False, default='wordle_5',
                        help='word dictionary')
    parser.add_argument('--plot', required=False, default=False, action='store_true', help='plotting or not')
    parser.add_argument('-p', '--previous', required=False, default='',
                        help='previous guesses from previous days to discard')

    args = parser.parse_args()
    
    guesses = args.guesses[0].split(',')
    rules = args.rules.split(',')
    previous = args.previous.split(',')
    
    assert len(guesses) == len(rules)
    
    words_by_dictionary, _ = wordle.load_dictionaries()
    remaining_words = words_by_dictionary[args.dictionary]
    
    if len(previous) > 0:
        remaining_words = list(set(remaining_words) - set(previous))  

#     for k in words_by_dictionary:
#         print(k, len(words_by_dictionary[k]))
#     print('dictionary: %s' % args.dictionary)

    print('# words in dictionary:', len(remaining_words))
    
    
    rules_encoded = []
    for i, w in enumerate(guesses):
        print('\nGUESS %i: %s' % (i, w))
        curr_rules = wordle.get_rules(w, rules[i])
        # print(curr_rules)
        remaining_words = wordle.select_multiple_rules(remaining_words, curr_rules)
        if len(remaining_words) < 30:
            print(remaining_words)

        rules_encoded += curr_rules
        print('# remaining words in dictionary:', len(remaining_words))
        # print(remaining_words)
    
    
    basename = str('-'.join([args.dictionary, "_".join(guesses), '_'.join(rules)]))
    out_basename = 'out/' + basename
    
    if int(args.plot):
        print('plotting')
        wordle.plot_words(remaining_words, out_basename=out_basename, plot=True)
        print('Heatmaps saved at out')
    
    guess = wordle.infer(remaining_words, args.strategy, sort_ascending=False)
    print('next guess (with strategy %s): %s' % (args.strategy, guess))
    
    if len(remaining_words) == 2:
        print('WARNING: There are only two words left. At this point, most strategies are guessing (50/50).')
