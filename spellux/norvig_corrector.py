#-*- coding: UTF-8 -*-

# Reuse of Peter Novig's spelling corrector prototype
# See https://norvig.com/spell-correct.html for the documentation

from __future__ import division
import os
import re
import math
import string
from fuzzywuzzy import fuzz
from collections import Counter

thedir = os.path.dirname(__file__)
data_dir = "data"

def words(text): return re.findall("[a-zA-Z-ëäöüéêèûîâÄÖÜËÉ'`-]+", text)

text_relpath = "rtl_news_articles_clean_puretext3.txt"
text_filepath = os.path.join(thedir, data_dir, text_relpath)
WORDS = Counter(words(open(text_filepath, encoding="utf-8").read()))

def P(word, N=sum(WORDS.values())):
    #Probability of word.
    return WORDS[word] / N

def correct(word):
    #Most probable spelling correction for word.
    return max(candidates(word), key=P)

def candidates(word):
    #Generate possible spelling corrections for word.
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    #The subset of `words` that appear in the dictionary of WORDS.
    return set(w for w in words if w in WORDS)

def edits1(word):
    #All edits that are one edit away from word.
    alphabet = "abcdefghijklmnopqrstuvwxyzëäöüéêèûîâABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜËÉ'-"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in alphabet]
    inserts    = [L + c + R               for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    #All edits that are two edits away from word.
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_text(text, sim_ratio):
    #Correct all the words within a text, returning the corrected text.
    corr_cand = re.sub("[a-zA-Z-ëäöüéêèûîâÄÖÜËÉ'`-]+", correct_match, text)
    if fuzz.ratio(text, corr_cand) >= sim_ratio:
        return corr_cand
    else:
        return text

def correct_match(match):
    #Spell-correct word in match, and preserve proper upper/lower/title case.
    word = match.group()
    return case_of(word)(correct(word))

def case_of(text):
    #Return the case-function appropriate for text: upper, lower, title, or just str.
    return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str)
