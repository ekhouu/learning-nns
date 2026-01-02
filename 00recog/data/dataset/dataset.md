# Dataset Info

## Data Sources

[French words dataset](https://raw.githubusercontent.com/Taknok/French-Wordlist/refs/heads/master/francais.txt) (old)
[French words dataset (new)](https://raw.githubusercontent.com/Thecoolsim/French-Scrabble-ODS8/refs/heads/main/French%20ODS%20dictionary.txt)
[German words dataset](https://gist.githubusercontent.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4/raw/36b70dd6be330aa61cd4d4cdfda6234dcb0b8784/wordlist-german.txt)
[World most common words dataset](https://github.com/frekwencja/most-common-words-multilingual)

## Assembly

1. Took all words in French or German out of global most common words dataset
2. Pruned words that appear in both French and German

if model is too stupid to deal with this, i will consider removing all germanic & romance languages, but hopefully not

I also used filter_german.py to remove a lot of German words because the original German wordlist dwarves the French one
