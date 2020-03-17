## spellux - Automatic text normalization for Luxembourgish

**Spellux** is a package made for the automatic correction of Luxembourgish texts. The package is written in Python 3. The code works but might still be a bit rough around the edges. The same holds true for correction accuracy: follow-up versions will bring an improved pretrained correction dictionary as well as additional algorithms for candidate evaluation. The resources in spellux are tailored to Luxembourgish, the architechture or or code snippets, however, might still be useful for other low-resource languages. So everyone feel free to test, contribute, modify.

In its current state, the package is a bit resource-heavy and not very fast, especially when you use it for larger chunks of text with a lot of words not yet recorded in the pretrained dictionary or lemma list. This will change with following releases and a better correction dictionary. For now, the package will be used mainly for testing, evaluation, and development, as it does return a number of wrong corrections. This is mainly due to the large amount of orthographic variation in Luxembourgish.

The current build produces some **false positive** corrections, i.e., for variants that match a lemma with a different meaning (e.g., *Sproch* 'saying' as a common spelling variant for *Sprooch* 'language'). It also produces some **false negative** corrections, i.e., wrong matches using the resource-based or string similarity matching modes (e.g., *gefaalt* 'folded' instead of *gefält* 'pleases' for *gefaelt* with Umlaut).

The package makes use of three pretrained correction resources:

- **Matching dictionary:** A pretrained dictionary with variant-lemma pairs trained on the comment data from RTL.lu. The dict is used for the first correction routine during processing. It can be updated with new word pairs and reset to retrain it on your own data.
- **Lemma list:** A list of lemmas from spellchecker.lu. The list is used for the second correction routine during processing.
- **Word embedding model:** A word embedding model (Word2Vec) trained on the entire RTL.lu corpus data. It is used for candidate evaluation in the third correction routine (modes 'model' and 'combo').

All correction resources are work in progress and will be updated with following releases. The matching dict can be expanded and modified by setting the *add_matches* parameter when calling normalization method and then calling the *update_resources()* method at the end of the training session. It can also be reset if you want to retrain it based on your own data set. 

The correction resources for the package stem from two major resources:

- **RTL.lu:** In the context of the STRIPS project at the Unviersity of Luxembourg (see [acc.uni.lu/strips/](https://acc.uni.lu/strips/)), the RTL Media Group has provided us with all article and comment text data published on the RTL.lu website between 2008 and 2018 for the development of semantic annotation algorithms (e.g., sentiment). See [rtl.lu](https://rtl.lu) for the news portal.
- **spellchecker.lu:** Michel Weimerskirch, author of the online correction tool [spellchecker.lu](https://spellchecker.lu), has provided us with a lemma list and variant-correction live data. See [github.com/spellchecker-lu/dictionary-lb-lu](https://github.com/spellchecker-lu/dictionary-lb-lu) for the resources.

### Installation

For now, you can download the repo to your harddisk. In Terminal, go to the top-level directory of the repo, then type:

```
pip install .
```

The package will be added to your local Python installation.

In order to prepare your system for working with spellux, run the following command to install all required dependencies:

```
pip install -r requirements.txt
```

The following external packages are required for working with spellux:

- **fuzzywuzzy** (>=0.17.0): fuzzy string matching using a modified Levenshtein distance
- **gensim** (>=3.8.0): processing of the word embedding model (Word2Vec)
- **progressbar** (>=2.5): progress visualization for text processing
- **spacy** (>=2.2.2): NLP including language support for Luxembourgish
- **scikit-learn** (>=0.22.1): machine learning package used for tf-idf matching


### Usage

To correct text with spellux, use the following basic commands:

```Python
# Import the package
import spellux

# Define dictionary with exceptions to override existing matches between variant and lemma
# This step is not obligatory
excs = {'Wort':'Wuert'}

# Define the text to match
# Note: This can be a string or a list of string tokens
text = "Eche hun d'Wort heut den Muaren mussen leesen."

# Call the function on a new instance
correct = spellux.normalize_text(text, exceptions=excs, mode='combo', sim_ratio=75, 
          add_matches=True, stats=True, print_unknown=False, nrule=True, indexing=False, 
          lemmatize=False, tolist=False, progress=False)

# Print the result of the correction
print(correct)
"Ech hu d'Wuert haut de Muere musse liesen."

# Save the updated matching dict and a list of unknown words to file
spellux.update_resources(matchdict=True, unknown=False, reset_matches=False)

# Print global correction statistics
spellux.global_stats(text)

# Use lemmatize method outside the main correction routine
# Note: this method takes a list of tokes as input, not a string
lemmas = spellux.lemmatize_text(text, sim_ratio=75)
print(lemmas)
```

Note: You can use the Jupyter notebook to test the package.

### Arguments of the normalize_text() method

The corrector function takes a couple of arguments to specify the processing and output options.

```Python
spellux.normalize_text(text, exceptions={}, mode='safe', sim_ratio=75, add_matches=True, 
                       stats=True, print_unknown=False, nrule=True, indexing=False, 
                       lemmatize=False, tolist=False, progress=False)
```

```Python
text
``` 
This is the only obligatory argument. It specifies the string or token list to process.

```Python
exceptions={}/dict_name
```
*Default setting: { }*

You can specify an exception dictionary (of type {*variant*:*lemma*}) to define specific variant-lemma pairs relevant to your data. 
**Note:** If you set the *add_matches* paramter to *True*, this option replaces the existing matches in the correction resources for the variants specified in the dict.

```Python
mode='safe'/'model'/'norvig'/'tf-idf'/'combo'
```
*Default setting: 'safe'*

Specify the correction mode for processing:

- **safe** mode uses only the existing correction resources (lemma list, pretrained matching dict) for correction. This increases the number of variants not found but limits the number of correction errors.
- **model** mode uses the word embedding model for candidate evaluation. The function returns the 10 nearest neighbors for a word and evaluates the most likely candidate using string simililarity matching in *fuzzywuzzy*. This mode is very fast but sometimes does not return a candidate, especially for rare variants.
- **norvig** mode uses the well-known spelling corrector written by Peter Norvig (see [norvig.com/spell-correct.html](https://norvig.com/spell-correct.html)). Here, words within max. 2 edits distance are evaluated by probability agaist a large text file (containing RTL atricle data). This method works best for typo detection. Also, it tends to be slow when processing long words.
- **tf-idf** mode uses an ngram-based similartiy matrix for candidate evaluation based on the *TfidfVectorizer* method in *scikit-learn*. Faster than *norvig* mode, and always returns a candidate – but sometimes weird ones.
- **combo** mode determines correction candidates based on a combination of *model*, *norvig*, and *td-idf*. The three candidates are evaluated using string simililarity matching in *fuzzywuzzy*. Given its composition of three correction processes plus evaluation, this is the slowest mode.

**Note:** There is an additional **training** mode that is only available internally for development and testing purposes. So don't bother activating it. Won't work.

```Python
sim_ratio=75 # number between 0 and 100
```
*Default setting: 75*

This setting specifies the similiaty ratio for candidate correction to finetune correction – and reduce the number of clearly wrong candidates. During the evaluation of a candiate for the correction, the input string and the correction candidates are compared as for string similarity using the *fuzz.ratio()* process in *fuzzywuzzy*. If the similarity ratio lies below the defined threshold, the correction candidate is disregarded. This setting affects the correction modes *'model'/'norvig'/'tf-idf'/'combo'*.

```Python
add_matches=False/True
```
*Default setting: True*

This setting allows you add new matches found during processing to a temporary extension of the matching dict. Set this to *False* when experimenting with the different correction modes.
**Note:** If you want to save the updated matching dict, call the *update_resources()* function at the end of a correction session.

```Python
stats=False/True
```
*Default setting: True*

This setting allows you to print basic statistics for the correction task:

- *Number of words:* The total number of words processed.
- *Number of new matches:* The number of new variant-lemma matches found during text processing.
- *Corrected items:* The number of items corrected during processing.
- *Items not found:* The number of items for which no correction was found in the correction resources.

```Python
print_unknown=False/True
```
*Default setting: False*

Setting this parameter to *True* returns a list of the items not found during the correction process.

```Python
nrule=False/True
```
*Default setting: True*

Option to correct the so called 'n-rule', a characteristic of Luxembourgish orthograpy based on phonetic context. Words ending in *-n* or *-nn* only keep their ending before vowel and certain consonants (n, t, d, z, h) in the onset of the following word. See [en.wikipedia.org/wiki/Eifeler_Regel#Luxembourgish](https://en.wikipedia.org/wiki/Eifeler_Regel#Luxembourgish) for details. For a full documentation of the rule (and its exceptions), see [https://portal.education.lu/zls/ORTHOGRAFIE](https://portal.education.lu/zls/ORTHOGRAFIE), point '6. D'n-Reegel'.
**Note:** Right now, the package covers the basic rule and most exceptions, including remaining n-endings before punctuation. It does not correct exceptions for personal and geographic names.

```Python
indexing=False/True
```
*Default setting: False*

Parameter to activate text output indexing for testing and development. The current version uses the following indices:

- **'0':** No candidate found during processing **[item unchanged]**
- **'1':** Item found in the matching dict **[item unchanged/corrected]**
- **'2':** item found in the lemma list **[item unchanged/corrected]**
- **'3':** candidate correction based on the chosen correction mode **[item corrected]**

```Python
lemmatize=False/True
```
*Default setting: False*

Parameter to reduce word forms to the lemma. This process is based on a lemma-inflection form dictionary and only works for words in the variant list at the moment.
**Note:** This parameter does not interact well with the *n-rule* correction activated. But then again, why would you combine those two anyways?

```Python
tolist=False/True
```
*Default setting: False*

With this option set to *True*, you can write the output of the correction process to a list of tokens instead of to a string.

```Python
progress=False/True
```
*Default setting: False*

This option enables a progress bar for the correction process. This can be useful for larger portions of text.
**Note:** If you use the function in a **for loop**, e.g., as part of a workflow that corrects multiple small instances of text, set this option to *False* to avoid an ever-restarting progress bar.

### Arguments of the lemmatize_text() method

The lemmatizer takes two arguments for reducing word forms to their lemma

```Python
spellux.lemmatize_text(text, sim_ratio=75)
```

```Python
text
``` 
This is the only obligatory argument. It specifies the input to process.
**Note:** The input text has to be a list of tokens, not a string.

```Python
sim_ratio=75 # number between 0 and 100
```
*Default setting: 75*

As for the correction routine, this setting specifies the similiaty ratio for candidate correction to finetune lemmatization – and reduce the number of wrong candidates.


### Arguments of the update_resources() method

The update function takes three arguments to specify the resources to update.

```Python
spellux.update_resources(matchdict=True, unknown=False, reset_matches=False)
```

```Python
matchdict=False/True
```
*Default setting: True*

Option to update the matching dictionary with new items (pairs of {*variant*:*lemma*}) found during text processing. This includes the exception dictionary if defined.

```Python
unknown=False/True
```
*Default setting: False*

Option to save the list of words not found during the correction process to a text file in your working directory.

```Python
reset_matches=False/True
```
*Default setting: False*

Option to reset the matching dictionary. This can be useful if you notice a lot of correction mistakes and want to retrain the dict on other/better data or resources.

### Arguments of the global_stats() method

The global stats method takes one obligatory argument.

```Python
spellux.global_stats(corpus)
```
```Python
corpus
``` 
Pass the variable that holds your corpus here to compute global correction stats.

**Note:** This works best when correcting multiple texts in a for loop.

### State of affairs & roadmap

Right now, the correction routine produces too many false positive and false negative corrections. Some of these can hardly be avoided using automatic word correction, i.e., due to the large amount of spelling variation in Luxembourgish and the high number of homographs this produces when checking against the lemma list. The correction resorurces and algorithms for candicate corrections need to be tested in detail and improved to reduce the number of false negative corrections.

Another way of improving matching accuracy and candidate evaluation could be the integration of a POS and word-context (trigrams) look-up during correction. Since tokenization in spellux is done with *spaCy*, POS information easy to implement for the tokens to be corrected. Plus, the lemma list from spellchecker.lu does hold POS information. Both resources use slightly different POS annotation systems, though. These need to be harmonized to implement this resource.

I also want to try and replace the word embedding model with a character-based representation of spelling variants using the **char2vec** approach (see [https://github.com/Lettria/Char2Vec](https://github.com/Lettria/Char2Vec)).

### Package info

```Python:
name='spellux',
version='0.1',
published='March 2020'
description='Automatic text normalization for Luxembourgish',
url='https://github.com/questoph/spellux',
author='Christoph Purschke',
contact='christoph@purschke.info',
reference='http://hdl.handle.net/10993/42807',
license='MIT'
```
