## Changelog

### v0.1.5 [16.03.2026]

- Fix syntax error in requirements.txt (duplicate `>=` operator for jellyfish dependency)
- Replace unsafe `eval()` with `ast.literal_eval()` when loading inflection dictionary
- Fix incorrect use of `is` instead of `==` for string comparison in output format check
- Replace bare `except` clauses with `except Exception` to avoid swallowing system signals
- Fix mutable default argument `exceptions={}` in `normalize_text` to prevent state leakage between calls
- Remove unused `import math` from norvig_corrector

### v0.1.4 [09.08.2020]

- Add option "remove_stopwords" to remove stop words from text during correction

### v0.1.3 [06.08.2020]

- Add function to check correction candidates determined with the word embedding model against the lemma list
- Add new pretrained matching dict based on the improed embedding model routine

### v0.1.2 [25.03.2020]

- Replace "tolist" parameter in normalizer function with "output" parameter (string, list, json)
- Output to json may include original word, correction, lemma, n-rule correction, correction index and text position index (depending on parameters)
- Performance improvements for lemmatizer function
- Add folder "workflows" to repository for example usage and application scripts
- Bug fixes and code cleanup

### v0.1.1 [24.03.2020]

- Replace fuzzywuzzy with jellyfish for fuzzy string matching (using Jaro-Winkler distance)
- Add license info for norvig corrector
- Bug fixes and code cleanup

### v0.1 [17.03.2020]

- First public development build
