## Instructions

To run the Syntactic Parser we need to provide it with 6 arguments.

* The training english corpus

* The test english corpus

* The training swedish corpus

* The test swedish corpus

* Number of training samples

* Number of epochs 

* Option -0 for perceptron parser/ -1 for neural networkor parser

* Option -0 for basic features/ -1 for advanced features

The application will automatically produce projectivized corpuses both for training and testing treebanks.

The projectivized corpus will be saved in the **data** folder while the scores will be saved in the **output** folder. 

An example execution is the following: 
```
    python3 app.py data/en-ud-train-projective.conllu data/en-ud-test.conllu data/sv_lines-ud-train.conllu data/sv_lines-ud-test.conllu 500 2 1 1
```

Or just run:
```
    python3 app.py
```

To run the program with its default settings.

* Use of the usual conllu files for training and testing from **data** folder

* 2 epochs

* 500 samples

* Neural Network parser

* Improved features
