## Instructions

To run the Syntactic Parser we need to provide it with 6 arguments.

* The training english corpus

* The test english corpus

* The training swedish corpus

* The test swedish corpus

* Number of training samples

* Number of epochs 

The application will automatically produce projectivized corpuses both for training and testing treebanks.

The projectivized corpus will be saved in the **data** folder while the predicted tags-trees and scores will be saved in the **output** folder. 

There is also a **scores** folder which contains the scores from our tests.

An example execution is the following: 
```
    python3 app.py data/en-ud-train-projective.conllu data/en-ud-dev.conllu data/sv_lines-ud-train.conllu data/sv_lines-ud-dev.conllu 4000 2
```
