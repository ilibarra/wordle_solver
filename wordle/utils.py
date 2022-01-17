
from sklearn.metrics import mutual_info_score
import unidecode
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.cluster.hierarchy as hc

def get_rules(query_word, target_or_code):
    i = 0
    rules = []
    for qi, ti in zip(query_word, target_or_code):
        if qi == ti or ti == '2':
            rules.append([qi, i, True])
        else:
            if qi in target_or_code or ti == '1':
                rules.append([qi, -1, True])
                rules.append([qi, i, False])
            else:
                rules.append([qi, -1, False])            
        i += 1
    return rules

def select_one_rule(words, rule):
    letter, position, mask = rule
    if mask:
        return [w for w in words if ((w[position] == letter) if position != -1 else letter in w)]
    return [w for w in words if ((w[position] != letter) if position != -1 else letter not in w)]

def select_multiple_rules(words, rules, log=False):
    if log:
        print('# query words', len(words))
    remaining_words = [w for w in words]
    for ri, r in enumerate(rules):
        remaining_words = select_one_rule(remaining_words, r)
    if log:
        print('after rule %s: %i words' % (r, len(remaining_words)))
    return remaining_words

def get_guesses(words, challenge_word, queries):
    words_by_guess = {}
    for wi, guess in enumerate(queries):
        # if wi % 100 == 0:
        #     print(wi)
        words_by_guess[guess] = select_multiple_rules(words, get_rules(guess, challenge_word))
    return words_by_guess

def score_guesses(words_by_guess):
    res  = []
    for gi, guess in enumerate(words_by_guess):
        #         if gi % 10 == 0:
        #             print(gi)
        remaining_words = words_by_guess[guess]
        # print(guess, len(remaining_words))
        
        med = np.nan
        if len(remaining_words) > 1:
            med = get_med_cluster_size(remaining_words)
        res.append([guess, len(remaining_words), med])
    # print('med', med)
    res = pd.DataFrame(res, columns=['guess', 'words.left', 'median.size'])
    res = res.sort_values('words.left')
    return res

def get_highest_freq_word(df, n_letters=4, use_covariation=True):
    freq = []
    
    # print(df.columns)
    # select the top 5 most frequent words, per position
    for position in range(n_letters):
        next_freq = pd.Series(df[position].values.flatten()).value_counts().sort_values()
        freq.append(next_freq.sort_values(ascending=False).head(6).index)

    # select three letters among the top 5 in each, with highest frequency and lowest overlap
    words = set(df['word'])
    
    total = []
    for next_comb in list(itertools.product(*freq)):
        # print(c)
        guess = "".join(next_comb)
        if not guess in words:
            continue
        
        # single column frequencies
        singles = []
        for i in range(n_letters):
            singles.append(df[i] == next_comb[i])
        scores_single = 0
        for s in singles:
            scores_single += np.sum(s)            
        
        # double column frequencies
        doubles = []
        for i, j in itertools.combinations(range(n_letters), r=2):
            # print(i, j)
            doubles.append(df[i] == next_comb[j])                    
        # filter words with double letter occurrences.
        # Single statement is to assures that each double count is counted only once. 
        overlap_doubles = doubles[0]
        for d in doubles:
            overlap_doubles = overlap_doubles | d
        scores_double = np.sum(overlap_doubles)
        
        
        scores_tuple = [scores_single, scores_double]

        # final score
        score = np.sum(scores_single)
        if use_covariation:
            score -= scores_double
    
    
        total.append([next_comb, score, guess, len(set(next_comb)), guess in words, [scores_tuple, [scores_single, scores_double]]])

    scores = pd.DataFrame(total, columns=['combination', 'score', 'guess', 'n.uniq', 'exists', 'score.tuple'])    
    scores = scores[scores['exists']]

    scores = scores.sort_values(['n.uniq', 'score'], ascending=[False, False])
    return scores

def infer(words, select_by, sort_ascending=False, last_guesses=set(), log=False):
    df = pd.DataFrame([[letter for letter in w] for w in words])
    # print(df.head())
    # print(df.columns)
    
    
    freq = pd.Series(df.values.flatten()).value_counts().sort_values()
    df['wordfreq'] = [sum([freq[c] for c in set(w)]) for w in words]
    df['word'] = words
    
    n_letters = len(words[0])

    best_guess = None
    # print('applying rule %s' % select_by)
    if select_by in {'posfreqcovar', 'posfreq'}:
        # print('posfreq...')TABOO 
        res = get_highest_freq_word(df, n_letters, use_covariation='covar' in select_by)            
        res = res[~res['guess'].isin(last_guesses)]
        if log:
            print(res.head(10))

        best_guess = res['guess'].values[0]
        queries = [best_guess]
    elif select_by == 'wordfreq':
        # print('wordfreq...')
        words_options = df.sort_values(select_by, ascending=sort_ascending)['word'][:1]
        queries = list(words_options) # + list(words_score)
        best_guess = queries[0]
        
    return best_guess

def plot_words(remaining_words, plot=False, out_basename=None, challenge_word=None):
    linkage = None
    if plot:
        dism = get_dism(remaining_words)
        linkage = hc.linkage(scipy.spatial.distance.squareform(dism), method='average')
        sns.clustermap(dism, row_linkage=linkage, col_linkage=linkage, figsize=[5, 5])
                       # xticklabels=None, yticklabels=None)

        if out_basename is not None:
            plt.savefig(out_basename + '_words_2d.png')
            plt.close()
            
    df_remaining = pd.DataFrame([[letter for letter in w] for w in remaining_words])
    df_remaining.index = ['w.%i' % i for i in range(df_remaining.shape[0])]
    df_remaining.columns = ['c.%i' % i for i in range(df_remaining.shape[1])]
    import numpy as np
    df_mask = pd.DataFrame(index=df_remaining.index)
    
    df_mask['challenge'] = 'white'
    if challenge_word is not None:
        df_mask['challenge'] = np.where(pd.Series(remaining_words) == challenge_word, 'black', 'white')
    df_colors = df_remaining.copy()
    letters = pd.Series(df_remaining.values.flatten()).value_counts().sort_values()
    cmap = {k: idx for idx, k in enumerate(letters.sample(letters.shape[0], random_state=500).index)}
    for c in df_colors:
        df_colors[c] = df_remaining[c].map(cmap).astype(int)

    if plot:
        df_dism = pd.DataFrame([[i, j, dist(a, b)] for i, a in enumerate(remaining_words) for j, b in enumerate(remaining_words)], columns=['i', 'j', 'd'])
        df_dism = df_dism.pivot('i', 'j', 'd')
        linkage = hc.linkage(scipy.spatial.distance.squareform(df_dism), method='average')
        g = sns.clustermap(df_colors, row_linkage=linkage, col_cluster=False, annot=df_remaining if df_remaining.shape[0] < 50 else None,
                       fmt='', cmap=sns.color_palette("Paired"), figsize=[4, 6], row_colors=df_mask) #  xticklabels=None, yticklabels=None)
        g.ax_heatmap.tick_params(left=False, bottom=False)

        if out_basename is not None:
            plt.savefig(out_basename + '_words_1d.png')
            plt.close()
    
    if not plot:
        plt.close()
    if plot == False and out_basename is not None:
        plt.show()


#### Clustering util functions (distance, mutual information, clustering)


def dist(a, b):
    assert len(a) == len(b)
    return len(a) - sum(ai == bi for ai, bi in zip(a, b))

def get_dism(words):
    entries = []
    for i, a in enumerate(words):
        for j, b in enumerate(words):
            if j < i:
                continue
            d = dist(a, b)
            entries.append([i, j, d])
            if i != j:
                entries.append([j, i, d])

    df_dism = pd.DataFrame(entries, columns=['i', 'j', 'd'])
    df_dism = df_dism.pivot('i', 'j', 'd')
    return df_dism

def get_mut_info(words):
    n_letters = len(words[0])
    df = pd.DataFrame([[letter for letter in w] for w in words])
    freq = pd.Series(df.values.flatten()).value_counts().sort_values()
    df['score'] = [sum([freq[c] for c in set(w)]) for w in words]
    df['word'] = words
    
    # calculate mutual information between words, iteratively
    letter_m = []
    for pi in range(n_letters):
        for pj in range(n_letters):
            if pj > pi:
                continue
            for a in freq.index:
                for b in freq.index:
                    v1, v2 = df['word'].str[pi].str.contains(a), df['word'].str[pj].str.contains(b)
                    mut_info = np.log10(mutual_info_score(v1, v2) + 1e-10)
                    # mut_info = mutual_info_score(v1, v2)
                    letter_m.append([a + str(pi), b + str(pj), mut_info])
                    letter_m.append([b + str(pj), a + str(pi), mut_info])

                    
    dism = pd.DataFrame(letter_m, columns=['a', 'b', 'mut.info'])
    dism['k'] = dism['a'] + dism['b']
    dism = dism.drop_duplicates('k')
    del dism['k']
    dism = dism.pivot('a', 'b', 'mut.info')
    return dism

# return the higher size of groups upon clustering and assessment at different thresholds
def get_scores(cg, df, min_thr=0, max_thr=200, step=10):
    res = []
    for thr in range(0, 200, 10):
        # print(thr)
        try:
            plt.close()
            den = scipy.cluster.hierarchy.dendrogram(cg.dendrogram_row.linkage,
                                                     labels = df.index,
                                                     color_threshold=thr / 100)
            plt.close()
            from collections import defaultdict


            def get_cluster_classes(den, label='ivl'):
                cluster_idxs = defaultdict(list)
                for c, pi in zip(den['color_list'], den['icoord']):
                    for leg in pi[1:3]:
                        i = (leg - 5.0) / 10.0
                        if abs(i - int(i)) < 1e-5:
                            cluster_idxs[c].append(int(i))

                cluster_classes = {}
                for c, l in cluster_idxs.items():
                    i_l = [den[label][i] for i in l]
                    cluster_classes[c] = i_l

                return cluster_classes

            clusters = get_cluster_classes(den);
            cluster = []
            for i in df.index:
                included=False
                for j in clusters.keys():
                    if i in clusters[j]:
                        cluster.append(j)
                        included=True
                if not included:
                    cluster.append(None)

            df["cluster"] = cluster

            med = pd.Series(cluster).value_counts().median()
            res.append([thr, med])
        except Exception:
            res.append([thr, np.nan])
            
    res = pd.DataFrame(res, columns=['thr', 'median'])
    return res.sort_values('median', ascending=True)


def get_med_cluster_size(words):
    df = pd.DataFrame([[letter for letter in w] for w in words])
    dism = get_dism(words)
    linkage = hc.linkage(sp.distance.squareform(dism), method='average')
    cg = sns.clustermap(dism, row_linkage=linkage, col_linkage=linkage)
    med = get_scores(cg, df)['median'].values[0]
    return med



import multiprocessing
from multiprocessing import Process
from multiprocessing import Manager
import threading
from threading import Thread
import pandas as pd
import numpy as np
import os
import random
import tempfile
from os.path import join

class ThreadingUtils:
    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def run(function, input_list, n_cores, log_each=None, log=False,
            input_list_kwargs=None):
        print(('run function %s with n_cores = %i' % (function, n_cores)))
        print(function)
        # print 'with input list of len'
        # print len(input_list)
        # print 'in groups of %d threads' % n_threads

        assert n_cores <= 20

        # the type of input_list has to be a list. If not
        # then it can a single element list and we cast it to list.
        if not isinstance(type(input_list[0]), type(list)):
            input_list = [[i] for i in input_list]

        n_groups = int(len(input_list) / n_cores + 1)
        # print 'n groups', n_groups

        n_done = 0
        
        print('# Total groups', n_groups)
        for group_i in range(n_groups):
            start, end = group_i * n_cores, (group_i + 1) * n_cores
            
            if group_i % 10 == 0:
                print('Current group %i (start/end)' % group_i, start, end)
            # print 'start', start, 'end', end

            threads = [None] * (end - start)
            for i, pi in enumerate(range(start, min(end, len(input_list)))):
                next_args = input_list[pi]
                next_kwargs = None if input_list_kwargs is None else input_list_kwargs[pi]
                if log:
                    print(next_args)
                # print next_kmer
                threads[i] = Process(target=function, args=next_args, kwargs=next_kwargs)
                # print 'starting process #', i
                threads[i].start()

            # print  threads
            # print 'joining threads...'
            # do some other stuff
            for i in range(len(threads)):
                if threads[i] is None:
                    continue
                threads[i].join()

                n_done += 1
                if log_each is not None and log_each % n_done == 0:
                    print('Done %i so far' % n_done)
        print('done...')
        
        
def load_dictionaries():
    # source
    # 'https://github.com/hannahcode/wordle/blob/main/src/constants/wordlist.ts'
    words_by_dictionary = {}
    df_by_dictionary = {}
    for n_letters in range(3, 6):
        for name, path in zip(['wordle', 'american'], ['data/wordle_list.txt', 'data/american-english']):
            words = [r.strip() for r in open(path)]
            words = [r.upper() for r in words if len(r) == 5 and not '\'' in r]
            words = [unidecode.unidecode(w) for w in words]
            if n_letters < 5:
                shorter_words = []
                for w in words:
                    for i in range(5 - n_letters + 1):
                        wi = w[i: i + n_letters]
                        shorter_words.append(wi)
                words = shorter_words

            words = list(set([w[:n_letters] for w in words])) # removing duplicates # in case reducing the dictionary complexity

            n_sample = None # 1000 # None # 1000 # None
            # random.seed()

            # print('# total words', len(words))
            if n_sample is not None:
                words = random.sample(words, n_sample)


            # print('# words (after sampling)', len(words))

            k = name + '_' + str(n_letters)
            # print(k)
            words_by_dictionary[k] = words
            # words = words[:n_sample]

            ### Prepare the dictionary containing the frequencies per position and mutual information of main dictionary
            ## step 1: prepare a mutual information table for our initial set of words
            df = pd.DataFrame([[letter for letter in w] for w in words])
            pd.Series(df.values.flatten()).value_counts().sort_values().plot(kind = 'barh')
            freq = pd.Series(df.values.flatten()).value_counts().sort_values()
            df['score'] = [sum([freq[c] for c in set(w)]) for w in words]
            df['word'] = words
            # print('preparing mutual information scores for smallest words...')
            # mut_info = get_mut_info(words)
            # plt.scatter(df['score'], df['mut.info.sum'])
            # df.sort_values(['score'], ascending=[False]).head(10)
            df_by_dictionary[k] = df
    return words_by_dictionary, df_by_dictionary
