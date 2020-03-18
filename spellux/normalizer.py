#-*- coding: UTF-8 -*-

# Spellux is a package for the automatic correction of Luxembourgish texts
# See https://github.com/questoph/spellux/ for the documentation

import re
import os
import csv
import string
import json
import operator
from collections import Counter
from fuzzywuzzy import process, fuzz
from gensim.models import Word2Vec
from spacy.lang.lb import Luxembourgish
nlp = Luxembourgish()
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from progressbar import ProgressBar
pbar = ProgressBar()

thedir = os.path.dirname(__file__)

from .norvig_corrector import correct_text

# Import correction resources
print("Preparing spellux resources:")
## Matching dict (variant:lemma) trained from comment data from RTL.lu
print("- importing matching dictionary")
match_dict = {}
matchdict_relpath = "data/matching_dict.txt"
matchdict_filepath = os.path.join(thedir, matchdict_relpath)
with open(matchdict_filepath, "r", encoding="utf-8") as match_file:
    data = csv.reader(match_file, delimiter=',')
    for row in data:
        match_dict[row[0]] = row[1]

## Lemma list based on data from spellchecker.lu
print("- importing lemma list")
lemlist_relpath = "data/lemma_list_spellchecker.txt"
lemlist_filepath = os.path.join(thedir, lemlist_relpath)
lemma_set = set(line.strip() for line in open(lemlist_filepath))
lemma_list = list(lemma_set)

## Lemma dictionary with variants for lemmatization
print("- importing inflection dictionary")
lemdict_relpath = "data/lemma_dict_pos.json"
lemdict_filepath = os.path.join(thedir, lemdict_relpath)
lemdict_json = open(lemdict_filepath, "r", encoding="utf-8").read()
lemdict = json.loads(lemdict_json)

## Word embedding model based on text data (articles, comments) from RTL.lu
print("- importing word embedding model")
model_relpath = "data/rtl_data_case_model_dim200_win5_iter5_count25.bin"
model_filepath = os.path.join(thedir, model_relpath)
model = Word2Vec.load(model_filepath)

## Additional lists for n-rule correction
print("- importing n-rule exception lists")
### List of words ending in nn to whom the n-rule applies
nns_relpath = "data/nn_replace_list.txt"
nns_filepath = os.path.join(thedir, nns_relpath)
nn_replace_list = set(line.strip() for line in open(nns_filepath))

### List of words ending in n to whom the n-rule does not apply
ns_relpath = "data/n_replace_list.txt"
ns_filepath = os.path.join(thedir, ns_relpath)
n_replace_list = set(line.strip() for line in open(ns_filepath))

## Train Tfidf matrix based on ngrams of words in lemma list
### Function to produce ngrams for words in list
def ngrams(string, n=2):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

print("- creating TF-IDF matrix")
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(lemma_set)
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)

## Variant/frequency dictionary based on data from spellchecker.lu. 
### Only available for internal training purposes
corrdict_relpath = "data/correction_dict_extended.json"
corrdict_filepath = os.path.join(thedir, corrdict_relpath)
with open(corrdict_filepath, encoding="utf-8") as corr_file:
    try:
        corr_dict = json.load(corr_file)
        print("- importing variant/frequency dictionary")

        lemvar_dict = {}
        lemma_freq = {}

        for lemma, entry in corr_dict.items():
            lemvar_dict[lemma] = set(entry["vars"])
            lemma_freq[lemma] = entry["freq"]
    except:
        pass

# Define list to collect unknown words
not_in_dict = set()  
savedir = os.getcwd()
unknown_relpath = "unknown_words.txt"
unknown_filepath = os.path.join(savedir, unknown_relpath)

print("\n All set. Let's go!")

# Function to print global test statistics
totals = {"words":0, "corrections":0, "misses":0}

def global_stats(corpus, totals=totals, reset=False, report=False):
    if isinstance(corpus, list) == True:
        texts = len(corpus)
    elif isinstance(corpus, str) == True:
        texts = 1
    if reset == True:
        totals["words"], totals["corrections"], totals["misses"] = 0, 0, 0
        print("Global stats have been set to zero!")
    else:
        if report == True:
            print("Correction statistics:\n")
            print("- Number of texts: {}" .format(str(texts)))
            print("- Number of words: {}" .format(str(totals["words"])))
            print("- Corrected items: {}" .format(str(totals["corrections"])))
            print("- Items not found: {}" .format(str(totals["misses"])))
        else:
            stats = [("texts", texts), ("words", totals["words"]), ("corrections", totals["corrections"]), ("unknown", totals["misses"])]
            return stats

# Correction functions for fuzzy string matching
## Function to evaluate candiate for correction using word embedding model
### Use fuzzy string matching to evaluate candidate from 10 nearest neighbors
def eval_emb_cand(word, sim_ratio):
    emb_cands = set()
    # Determine embedding candidates in model
    try:
        sim_cands = model.wv.most_similar(positive=word, topn=10)
        for cand in sim_cands:
            emb_cands.add(cand[0])
        # Evaluate correction candidate using fuzzy string matching  
        fuzz_cand = process.extractOne(word, emb_cands)
        emb_cand = fuzz_cand[0]
        if fuzz.ratio(word, emb_cand) >= sim_ratio:
            return emb_cand
        else:
            return word
    except KeyError:
        return word
  
## Function to evaluate the K nearest neighbors using TF-IDF
def eval_lem_cand(word, lemmalist, sim_ratio):
    if isinstance(word, list) == False:
        word = [word]
    queryTFIDF_ = vectorizer.transform(word)
    list_index = nbrs.kneighbors(queryTFIDF_)
    lem_cand = "".join(lemmalist[int(list_index[1])])
    if fuzz.ratio(word, lem_cand) >= sim_ratio:
        return lem_cand
    else:
        word = "".join(word)
        return word
    
## Function to evaluate candiatate based on combination of other methods
def eval_combo_cand(word, lemmalist, sim_ratio):
    train_cands = [eval_emb_cand(word, sim_ratio), correct_text(word, sim_ratio), eval_lem_cand(word, lemmalist, sim_ratio)]
    counts = Counter(train_cands)
    maxval = max(counts.values())
    if maxval >= 2:
        cand = "".join([k for k, v in counts.items() if v == maxval])
        return cand
    else:
        extract_cand = process.extractOne(word, train_cands)
        combo_cand = extract_cand[0]
        if fuzz.ratio(word, combo_cand) >= sim_ratio:
            return combo_cand
        else:
            return word

## Function to check word against variant/frequency dict:
### Only available internally for training the matching dictionary
def eval_varfreq_cand(word, lemvardict, lemfreq):
    corr_vars = []
    for k, v in lemvardict.items(): 
        if word in v:
            corr_vars.append(k)
    # Replace word if 1 variant exists in correction dict
    if len(corr_vars) == 0: 
        cand = word 
    elif len(corr_vars) == 1: 
        cand = "".join(corr_vars)
    elif len(corr_vars) > 1:
        freq_corr = {}
        for el in corr_vars:
            for lemma,freq in lemfreq.items():
                if el == lemma:
                    freq_corr[el] = freq
                else:
                    pass
        # Evaluate correction candidates based on max value for frequency
        cand = max(freq_corr.items(), key=operator.itemgetter(1))[0]
    return cand
        
## Function for the correction of n-rule spellings 
### Covers basic rule including most exceptions; 
### Foreign words and names handling still missing)
def correct_nrule(text, indexing):
    correct_text = []
    # List of onset context for which predecessing n is not deleted
    context_list = ["u", "U", "n", "N", "i", "I", "t", "T", "e", "E", "d", "D", "z", "Z", "o", "O", "h", "H", "a", "A", "é", "É", "ë", "Ë", "ä", "Ä", "ö", "Ö", "ü", "Ü", ".", ",", "!", "?", ";", "(", ":", "-", "1", "2", "3", "8", "9"]
    # List of onset+1 contexts for which n is not deleted before y
    context_y = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "z"]
    # Copy of text without indices for index matching in indexing mode
    text_ = [] 
    for token in text:
        if token in string.punctuation:
            text_.append(token)
        else:
            text_.append(token[:-1])
    # Set index buffer for context look-up
    index_buffer = {}
    # start correction routine
    for word in text:
        if indexing == True:
            text = text_
            if word in string.punctuation:
                index = ""
            else:
                index = word[-1]
                word = word[:-1]
        elif indexing == False:
            index = ""
        # Create image of text with index position for every word
        buff = 0
        if word in index_buffer:
            buff = index_buffer[word] +1
        lem_index = text.index(word, buff)
        context_right = "".join(text[lem_index+1:lem_index+2])
        index_buffer[word] = lem_index
        if word in string.punctuation:
            correct_text.append(word)
        elif word[-2:] == "nn":
            # Strip "-nn" ending from word
            if word in nn_replace_list:
                try:
                    onset = context_right[0]
                    yonset = context_right[1]
                    # unitedzoha rule
                    if onset in context_list:
                        correct_text.append(word + index)
                    # y + consonant exception
                    elif onset.lower() == "y" and yonset in context_y:
                        correct_text.append(word + index)
                    else:
                        correct_text.append(word[0:-2] + index)
                except IndexError:
                    correct_text.append(word + index)
            else: 
                correct_text.append(word + index)
        elif word[-1] == "n":
            # Strip "-n" ending from word            
            if word in n_replace_list or word.endswith("en") == True or word.endswith("äin") == True or word.endswith("eeën") == True:
                try:
                    onset = context_right[0]
                    yonset = context_right[1]
                    # unitedzoha rule
                    if onset in context_list:
                        correct_text.append(word + index)
                    # y + consonant exception
                    elif onset == "Y" and yonset in context_y:
                        correct_text.append(word + index)
                    elif onset == "y" and yonset in context_y:
                        correct_text.append(word + index)
                    else:
                        correct_text.append(word[0:-1] + index)
                except IndexError:
                    correct_text.append(word + index)
            else:
                correct_text.append(word + index)
        else:
            correct_text.append(word + index)
    return correct_text
    
## Function to reduce word forms to their lemma 
### Based on the inflection for dictionary
def lemmatize_text(text, lemdict=lemdict, indexing=False, sim_ratio=75):
    correct_text = []
    text_ = [] 
    pattern = r"^[a-zA-Z]([\w'`-])*[a-zA-Z]?$"
    for token in text:
        if token in string.punctuation or re.match(pattern, token) is None:
            text_.append(token)
        else:
            text_.append(token[:-1])
    # start correction routine
    for word in text:
        if indexing == True:
            text = text_
            if word in string.punctuation or re.match(pattern, token) is None:
                index = ""
            else:
                index = word[-1]
                word = word[:-1]
        elif indexing == False:
            index = ""
        lem_cands = []
        for k, v in lemdict.items(): 
            if word in v["variants"]:
                tup = (k, v["pos"])
                lem_cands.append(tup)
        # Replace word if 1 variant exists in correction dict
        if len(lem_cands) == 0: 
            correct_text.append(word + index) 
        elif len(lem_cands) == 1: 
            cand = lem_cands[0][0]
            correct_text.append(cand + index)
        elif len(lem_cands) > 1:
            pos_check = [lemtup[1] for lemtup in lem_cands]
            # Map derived adjectives to original word type
            if len(lem_cands) == 2 and "ADJ" in pos_check:
                cand = "".join([lemtup[0] for lemtup in lem_cands if not lemtup[1] == "ADJ"])
                correct_text.append(cand + index)
            else:
                # Extract likeliest candidate using fuzzy string matching
                cands = [lemtup[0] for lemtup in lem_cands]
                extract_cand = process.extractOne(word, cands)
                lem_cand = extract_cand[0]
                if fuzz.ratio(word, lem_cand) >= sim_ratio:
                    correct_text.append(lem_cand + index)
                else:
                    correct_text.append(word + index)
    return correct_text

# Functions to save the updated matchdict and a list of unknown words to file
## Function to write the updated matching dict to file or reset it to empty
def update_matchdict(matchdict, reset):
    with open(matchdict_filepath, "w", newline="", encoding="utf-8") as out:
        if reset == False:
            for key in matchdict.keys():
                out.write("{},{}\n" .format(key, matchdict[key]))
        elif reset == True:
            matchdict = ""
            out.write(matchdict)
            
## Function to save unknown words to file
def save_unknown(notindict):
    with open(unknown_filepath, 'w', encoding="utf-8") as out:
        for item in notindict:
            out.write("{}\n" .format(item))

## Function to update resources
def update_resources(matchdict=True, unknown=False,  reset_matchdict=False):
    if matchdict == True:
        update_matchdict(match_dict, reset_matchdict)
    if unknown == True:
        save_unknown(list(not_in_dict))

# Main function to correct text based on correction resources
## Set options to streamline workflow
def normalize_text(text, matchdict=match_dict, exceptions={}, mode="safe", sim_ratio=75, add_matches=True, stats=True, nrule=True, print_unknown=False, indexing=False, lemmatize=False, tolist=False, progress=False):
    
    #Set alphabet for string pattern matching
    alpha = "a-zA-Z-äüöÄÖÜëéËÉ"
    
    # Define set to collect unknown words
    not_found = set()
    
    # Set counters for stats
    word_count, corr_count, miss_count = 0, 0, 0
    
    # Include exception dict in matching dict
    if len(exceptions) > 0:
        match_dict.update(exceptions)
    
    match_count = len(match_dict)
    
    # Parse input text from string or token list
    if isinstance(text, list) == True:
        words = text
    else:
        # Tokenize string input with spaCy
        words = []
        doc = nlp(text)
        for token in doc:
            words.append(token.text)
    
    if progress == True:
        # Set progress bar for correction process
        words_ = pbar(words).start()
    else:
        words_ = words
    text_corr = []
    # Start correction routine
    for word in words_:
        word_count +=1
        if word in string.punctuation:
            # Keep punctuation unchanged
            text_corr.append(word)
            word_count -=1
        elif re.match(rf"^[{alpha}]([{alpha}'`-])*[{alpha}]?$", word) is None:
            # Keep non-alphabetic words & special characters unchanged
            text_corr.append(word)
            word_count -=1
        elif word in match_dict:
            # Check word against matching dict
            repl = match_dict[word]
            if word.istitle() == True and repl.islower() == True:
                repl = repl.title()
            if indexing == True:
                text_corr.append(repl + "₁")
            else:
                text_corr.append(repl)
            if word != repl:
                corr_count +=1
        elif word in lemma_set:
            # Check word against lemma list
            if indexing == True:
                text_corr.append(word + "₂")
            else:
                text_corr.append(word)
            if add_matches == True:
                match_dict[word] = word
        elif word.lower() in lemma_set:
            # Check word.lower() against lemma list
            if indexing == True: 
                text_corr.append(word + "₂")
            else:
                text_corr.append(word)
            if add_matches == True:
                match_dict[word] = word.lower()
        elif word.title() in lemma_set:
            # Check word.title() against lemma list
            if indexing == True:
                text_corr.append(word_.title() + "₂")
            else:
                text_corr.append(word.title())
            if add_matches == True:
                match_dict[word] = word.title()
            corr_count +=1
        else: 
            # Stop correction here if set to safemode (no fuzzy matching)
            if mode == "safe":
                if indexing == True:
                    text_corr.append(word + "₀")
                else:
                    text_corr.append(word)
                not_found.add(word)
                miss_count +=1
            else:
                if mode == "model": 
                    # Evaluate correction candidate using word embedding model
                    deamb = eval_emb_cand(word, sim_ratio)
                elif mode == "norvig":
                    # Evaluate correction candidate using norvig corrector
                    deamb = correct_text(word, sim_ratio)
                elif mode == "tf-idf":
                    # Evaluate correction candidate using learned tf-idf matrix
                    deamb = eval_lem_cand(word, lemma_list, sim_ratio)
                elif mode == "combo":
                    # Evaluate correction candidate using all three resources
                    deamb = eval_combo_cand(word, lemma_list, sim_ratio)
                elif mode == "training":
                    # Evaluate correction candidate using word variant/frequency dict
                    try:
                        deamb = eval_varfreq_cand(word, lemvar_dict, lemma_freq)
                    except:
                        print("Mode 'training' not available. Please use another mode.")
                        break
                else:
                    print("This is not a valid mode! Please try again.")
                    break
                
                if deamb == word:
                    # Keep uncorrected word, add it to list of missing words
                    if indexing == True:
                        text_corr.append(word + "₀")
                    else:
                        text_corr.append(word)
                    not_found.add(word)
                    miss_count +=1
                else:
                    # Add evaluated correction candidate
                    if indexing == True: 
                        text_corr.append(deamb + "₃")
                    else:
                        text_corr.append(deamb)
                    if add_matches == True:
                        match_dict[word] = deamb
                    corr_count +=1
    # Add words not found to not_in_dict
    not_in_dict.update(not_found)
    # Add counts to global stats
    totals["words"] += word_count
    totals["corrections"] += corr_count
    totals["misses"] += miss_count
    # Call function to correct n-rule if set to True
    if nrule == True:
        text_corr = correct_nrule(text_corr, indexing)
    # Lemmatize words if set to True
    if lemmatize == True:
        text_corr = lemmatize_text(text_corr, lemdict, indexing, sim_ratio)
    # Print statistics if set to True
    if stats == True:
        print("Number of words: {}" .format(str(word_count)))
        print("Number of new matches: {}" .format(str(len(match_dict) - match_count)))
        print("Corrected items: {}" .format(str(corr_count)))
        print("Items not found: {}" .format(str(miss_count)))
    # Print list of items not found if set to True
    if print_unknown == True:
        print(list(not_found))
    # Print list of corrected tokens (True) or re-join text to string
    if tolist == True:
       pass 
    else:
        text_corr = " ".join(text_corr)
        # Strip leading whitespaces for punctiation
        text_corr = re.sub(r'\s([?.,:;!"](?:\s|$))', r'\1', text_corr)
        # Trailing whitespaces for items of type "d'"
        d_list = ["d' ", "D' ", "d` ", "D` "]
        for d in d_list:
            if d in text_corr:
                text_corr = text_corr.replace(d, d.strip())
    return text_corr
