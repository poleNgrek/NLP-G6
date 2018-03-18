import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
import DNN

class Parser(object):

    def __init__(self):
        """Initialises a new parser."""
        import Tagger
        self.tagger = Tagger.Tagger()
        #self.classifier = Tagger.Perceptron()
        def one_hot_encode_object_array(arr):
            uniques, ids = np.unique(arr, return_inverse=True)
            return uniques,np_utils.to_categorical(ids, len(uniques))
        self.uniques,self.number=one_hot_encode_object_array([0,1,2])
        self.classifier=DNN.Classifier(self.uniques,6)

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
            self.classifier.update(features, self.number[gold])
            #self.classifier.update(features, gold)
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

        if i >= len(words):
            features.append('<EOS>')
            features.append('<EOS>')
        else:
            features.append(words[i])
            features.append(tags[i])

        if len(stack) > 0:
            features.append(words[stack[-1]])
            features.append(tags[stack[-1]])
        else:
            features.append('<EMPTY>')
            features.append('<EMPTY>')

        if len(stack) > 1:
            features.append(words[stack[-2]])
            features.append(tags[stack[-2]])
        else:
            features.append('<EMPTY>')
            features.append('<EMPTY>')

        '''
        if len(stack) > 0:
            distance = i - stack[-1]
            leftW = '<EMPTY>'
            leftP = '<EMPTY>'
            for k in range(1, stack[-1]):
                if parse[k] == stack[-1]:
                    leftW = words[k]
                    leftP = tags[k]
                    break
            rightW = '<EMPTY>'
            rightP = '<EMPTY>'
            for k in range(i-1, stack[-1], -1):
                if parse[k] == stack[-1]:
                    rightW = words[k]
                    rightP = tags[k]
                    break
            if len(parse) > i:
                features += [(tags[stack[-1]], leftP, tags[i])]
                features += [(words[stack[-1]], leftP, tags[i])]
                features += [(tags[stack[-1]], leftW, tags[i])]
                features += [(leftW)]
            else:
                features += [(tags[stack[-1]], leftP, '<EMPTY>')]
                features += [(words[stack[-1]], leftP, '<EMPTY>')]
                features += [(tags[stack[-1]], leftW, '<EMPTY>')]
                features += [(leftW)]
        else:
            if len(parse) > i:
                # 23: S0pS0lpN0p
                features += [( '<EOS>', '<EMPTY>', tags[i])]
                # 23.1: S0wS0lpN0p
                features += [( '<EOS>', '<EMPTY>', tags[i])]
                # 23.2: S0pS0lwN0p
                features += [( '<EOS>', '<EMPTY>', tags[i])]
                # 23.3: S0lw
                features += ['<EMPTY>']
            else:
                # 23: S0pS0lpN0p
                features += [('<EOS>', '<EMPTY>', '<EMPTY>')]
                # 23: S0wS0lpN0p
                features += [('<EOS>', '<EMPTY>', '<EMPTY>')]
                # 23.2: S0pS0lwN0p
                features += [('<EOS>', '<EMPTY>', '<EMPTY>')]
                # 23.3: S0lw
                features += ['<EMPTY>']
        '''

        return list(zip(range(len(features)), features))

    def finalize(self):
        #self.classifier.finalize()
        self.tagger.finalize()