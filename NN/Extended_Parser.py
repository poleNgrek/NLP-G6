import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
import DNN

class Parser(object):

    def __init__(self, option):
        """Initialises a new parser."""
        import Tagger
        print(option)
        self.tagger = Tagger.Tagger()
        self.option = int(option)

        if self.option == 1:
            self.classifier = Tagger.Perceptron()
        else:
            def one_hot_encode_object_array(arr):
                uniques, ids = np.unique(arr, return_inverse=True)
                return uniques,np_utils.to_categorical(ids, len(uniques))
            
            self.uniques,self.number=one_hot_encode_object_array([0,1,2])
            self.classifier=DNN.Classifier(self.uniques,10)

    def parse(self, words):
        pred_tree = [0] * len(words)
        i = 0
        stack = []

        taggisar = self.tagger.tag(words)

        while True:
            valid = self.valid_moves(i, stack, pred_tree)
            if not valid:
                break
            features = self.features(words, taggisar, i, stack, pred_tree)

            move = self.classifier.predict(features, valid)

            i, stack, pred_tree = self.move(i, stack, pred_tree, move)

        return (taggisar, pred_tree)

    def valid_moves(self, i, stack, pred_tree):
        moves = []
        if i < len(pred_tree):
            moves.append(0)
        if len(stack) >= 3:
            moves.append(1)
        if len(stack) >= 2:
            moves.append(2)

        return moves

    def move(self, i, stack, pred_tree, move):
        if move == 0:
            stack.append(i)
            i += 1

        elif move == 1:
            pred_tree[stack.pop(-2)] = stack[-1]

        elif move == 2:
            head = stack[-2]
            pred_tree[stack.pop(-1)] = head

        return (i, stack, pred_tree)

    def update(self, words, gold_tags, gold_tree):
        stack = []
        pred_tree = [0] * len(words)
        i = 0

        tags=self.tagger.update(words, gold_tags)

        while True:
            valid = self.valid_moves(i, stack, pred_tree)
            if not valid:
                break

            gold = self.gold_move(i, stack, pred_tree, gold_tree)
            features = self.features(words, tags, i, stack, pred_tree)
            if self.option == 2:
                self.classifier.update(features, self.number[gold])
            if self.option == 1:
                self.classifier.update(features, gold)
            i, stack, pred_tree = self.move(i, stack, pred_tree, gold)

        return (tags,pred_tree)

    def gold_move(self, i, stack, pred_tree, gold_tree):
        valid = self.valid_moves(i, stack, pred_tree)
        if 1 in valid and stack[-1] == gold_tree[stack[-2]]:
            for j, e in enumerate(gold_tree):
                if e == stack[-2]:
                    if not pred_tree[j]:
                        break
            else:
                return 1

        if 2 in valid and stack[-2] == gold_tree[stack[-1]]:
            for j, e in enumerate(gold_tree):
                if e == stack[-1]:
                    if not pred_tree[j]:
                        break
            else:
                return 2

        if 0 in valid:
            return 0

        return

    def features(self, words, tags, i, stack, parse):

        features = []

        EMP = "<EMPTY>"
        EOS = "<EOS>"

        if len(parse) > i:
            features += [(1, words[i])]
            features += [(2, tags[i])]
        else:
            features += [(1, EOS)]
            features += [(2, EOS)]

        if len(stack) < 2:
            features += [(5, EMP)]
            features += [(6, EMP)]

            if len(stack) < 1:
                features += [(3, EMP)]
                features += [(4, EMP)]
            else:
                features += [(3, words[stack[-1]])]
                features += [(4, tags[stack[-1]])]
        else:
            features += [(3, words[stack[-1]])]
            features += [(4, tags[stack[-1]])]

            features += [(5, words[stack[-2]])]
            features += [(6, tags[stack[-2]])]


        if len(stack) > 0:
            distance = i - stack[-1]
            leftW = EMP
            leftP = EMP
            for k in range(1, stack[-1]):
                if parse[k] == stack[-1]:
                    leftW = words[k]
                    leftP = tags[k]
                    break

            if len(parse) > i:
                features += [(7, tags[stack[-1]], leftP, tags[i])]
                features += [(8, words[stack[-1]], leftP, tags[i])]
                features += [(9, tags[stack[-1]], leftW, tags[i])]
                features += [(10, leftW)]
            else:
                features += [(7, tags[stack[-1]], leftP, EMP)]
                features += [(8, words[stack[-1]], leftP, EMP)]
                features += [(9, tags[stack[-1]], leftW, EMP)]
                features += [(10, leftW)]
        else:
            if len(parse) > i:
                features += [(7, EOS, EMP, tags[i])]
                features += [(8, EOS, EMP, tags[i])]
                features += [(9, EOS, EMP, tags[i])]
                features += [(10, EMP)]
            else:
                features += [(7, EOS, EMP, EMP)]
                features += [(8, EOS, EMP, EMP)]
                features += [(9, EOS, EMP, EMP)]
                features += [(10, EMP)]        

        return list(zip(range(len(features)), features))

    def finalize(self):
        if self.option == 1:
            self.classifier.finalize()
        self.tagger.finalize()
