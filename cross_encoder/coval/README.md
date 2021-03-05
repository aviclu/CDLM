# CoVal: A coreference evaluation tool for the CoNLL and ARRAU datasets

Implementation of the common evaluation metrics including MUC,
B-cubed, CEAFe, and LEA for both CoNLL and ARRAU datasets.
See the paper [Which Coreference Evaluation Metric Do You Trust? A Proposal for
a Link-based Entity Aware Metric](https://www.aclweb.org/anthology/P16-1060).

### Requirements
This evaluation tool requires numpy and scipy packages.

## Usage
Basic usage with CoNLL files:

	$ python scorer.py key system

`key` and `system` are the files with gold coreference and system output, respectively.

For more details, refer to
[ARRAU README](https://github.com/ns-moosavi/coval/blob/master/arrau/README.md)
for evaluations of the ARRAU files and
[CoNLL README](https://github.com/ns-moosavi/coval/blob/master/conll/README.md)
for CoNLL evaluations.

Run tests with `python3 -m pytest unittests.py`

## Reference
If you use this code in your work, please cite the paper:
```
@InProceedings{moosavi2019minimum,
    author = { Nafise Sadat Moosavi, Leo Born, Massimo Poesio and Michael Strube},
    title = {Using Automatically Extracted Minimum Spans to Disentangle Coreference Evaluation from Boundary Detection},
    year = {2019},
    booktitle = {Proceedings of the 57th Annual Meeting of
		the Association for Computational Linguistics (Volume 1: Long Papers)},
    publisher = {Association for Computational Linguistics},
    address = {Florence, Italy},
}
```

## Authors
This code was written by [@ns-moosavi](https://github.com/ns-moosavi/).
Some parts are borrowed from
https://github.com/clarkkev/deep-coref/blob/master/evaluation.py

The test suite is taken from https://github.com/conll/reference-coreference-scorers/

Mention evaluation and the test suite are added by
[@andreasvc](https://github.com/andreasvc/).

Parsing CoNLL files is developed by Leo Born.
