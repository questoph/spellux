## Changelog

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
