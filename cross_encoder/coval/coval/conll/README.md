
# Using CoVal for evaluating the CoNLL dataset



## Usage


The following command evaluates coreference outputs related to the CoNLL dataset:


python scorer.py key system [options]


'key' and 'system' are the key and system files, respectively.



## Evaluation Metrics


The above command reports MUC [Vilain et al, 1995], B-cubed [Bagga and Baldwin, 1998], CEAFe [Luo et al., 2005], LEA [Moosavi and Strube, 2016] and the averaged CoNLL score (the average of the F1 values of MUC, B-cubed and CEAFe) [Denis and Baldridge, 2009a; Pradhan  et  al., 2011].


You can also only select specific metrics by including one or some of the 'muc', 'bcub', 'ceafe' and 'lea' options in the input arguments.

For instance, the following command only reports the CEAFe and LEA scores:


python scorer.py key system ceafe lea


The first and second arguments after 'scorer.py' have to be 'key' and 'system', respectively. The order of other options is arbitrary.


## Evaluation Modes


1) With or without singletons


After extracting all mentions of key or system files, mentions whose corresponding coreference chain is of size one, are considered as singletons.

The default evaluation mode will include singletons in evaluations if they are included in the key or the system files.

By including the 'remove_singletons' or 'remove_singleton' options, all singletons in the key and system files will be excluded from the evaluation.


2) NP only


Most of the recent coreference resolvers only resolve NP mentions and leave out the resolution of VPs.
By including the 'NP_only' option, the scorer will only evaluate the resolution of NPs.

3) Using minimum spans

By including any of the 'min', 'min_span', or 'min_spans' options, the scorer reports the results based on automatically detected minimum spans.
Minimum spans are determined using the MINA algorithm.


