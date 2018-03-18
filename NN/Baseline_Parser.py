class Parser(object):
    """A transition-based dependency parser.
    
    This parser implements the arc-standard algorithm for dependency
    parsing. When being presented with an input sentence, it first
    tags the sentence for parts of speech, and then uses a multi-class
    perceptron classifier to predict a sequence of *moves*
    (transitions) that construct a dependency tree for the input
    sentence. Moves are encoded as integers as follows:
    
    SHIFT = 0, LEFT-ARC = 1, RIGHT-ARC = 2
    
    At any given point in the predicted sequence, the state of the
    parser can be specified by: the index of the first word in the
    input sentence that the parser has not yet started to process; a
    stack holding the indices of those words that are currently being
    processed; and a partial dependency tree, represented as a list of
    indices such that `tree[i]` gives the index of the head (parent
    node) of the word at position `i`, or 0 in case the corresponding
    word has not yet been assigned a head.
    
    Attributes:
        tagger: A part-of-speech tagger.
        classifier: A multi-class perceptron classifier used to
            predict the next move of the parser.
    """
    
    SH = 0
    LA = 1
    RA = 2
    
    def __init__(self):
        """Initialises a new parser."""
        import Tagger
        self.tagger = Tagger.Tagger()
        self.classifier = Tagger.Perceptron()
    
    def parse(self, words):
        """Parses a sentence.
        
        Args:
            words: The input sentence, a list of words.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        tags = self.tagger.tag(words)
        
        i = 1
        stack = [0]
        tree = [0] * len(words)
        
        while True:
            valid = self.valid_moves(i, stack, tree)
            if len(valid) < 1:
                break
            feat = self.features(words, tags, i, stack, tree)
            move = self.classifier.predict(feat, valid)
            i, stack, tree = self.move(i, stack, tree, move)
        
        return (tags, tree)
    
    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        valid = []
        if len(pred_tree) > i:
            valid += [self.SH]
        elif(len(stack) == 2):
            valid += [self.RA]
        if(len(stack) > 2):
            valid += [self.RA, self.LA]
        
        return valid
    
    def move(self, i, stack, pred_tree, move):
        """Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """
        
        if move == self.SH:
            stack += [i]
            i += 1
        elif move == self.LA:
            pred_tree[stack[-2]] = stack[-1]
            del stack[-2]
        elif move == self.RA:
            pred_tree[stack[-1]] = stack[-2]
            del stack[-1]
        
        return (i, stack, pred_tree)
        
    
    def update(self, words, gold_tags, gold_tree):
        """Updates the move classifier with a single training
        instance.
        
        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input
                sentence.
            gold_tree: The gold-standard tree for the sentence.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        
        tags = self.tagger.update(words, gold_tags)
        
        i = 0
        stack = []
        tree = [0] * len(words)
        
        while True:
            move = self.gold_move(i, stack, tree, gold_tree)
            feat = self.features(words, tags, i, stack, tree)
            self.classifier.update(feat, move)
            i, stack, tree = self.move(i, stack, tree, move)
            if i > 1 and stack == [0]:
                break
        
        return (tags, tree)
    
    def gold_move(self, i, stack, pred_tree, gold_tree):
        """Returns the gold-standard move for the specified parser
        configuration.
        
        The gold-standard move is the first possible move from the
        following list: LEFT-ARC, RIGHT-ARC, SHIFT. LEFT-ARC is
        possible if the topmost word on the stack is the gold-standard
        head of the second-topmost word, and all words that have the
        second-topmost word on the stack as their gold-standard head
        have already been assigned their head in the predicted tree.
        Symmetric conditions apply to RIGHT-ARC. SHIFT is possible if
        at least one word in the input sentence still requires
        processing.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            gold_tree: The gold-standard dependency tree.
        
        Returns:
            The gold-standard move for the specified parser
            configuration, or `None` if no move is possible.
        """
        if len(stack) < 2:
            if len(pred_tree) > i:
                return self.SH
            return None
        
        if gold_tree[stack[-2]] == stack[-1]:
            left = True
            for k in range(len(gold_tree)):
                if gold_tree[k] == stack[-2] and pred_tree[k] == 0:
                    left = False
            if left:
                return self.LA
            
        if gold_tree[stack[-1]] == stack[-2]:
            right = True
            for k in range(len(gold_tree)):
                if gold_tree[k] == stack[-1] and pred_tree[k] == 0:
                    right = False
            if right:
                return self.RA
        
        if len(pred_tree) > i:
            return self.SH
        return None
    
    def features(self, words, tags, i, stack, parse):
        """Extracts features for the specified parser configuration.

        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input
                sentence.
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            parse: The partial dependency tree.

        Returns:
            A feature vector for the specified configuration.
        """
        EMP = "<EMPTY>"
        EOS = "<EOS>"
        
        if len(parse) > i:
            feat = [(0, words[i])]
            feat += [(1, tags[i])]
        else:
            feat = [(0, EOS)]
            feat += [(1, EOS)]
        
        if len(stack) < 2:
            feat += [(4, EMP)]
            feat += [(5, EMP)]
            
            if len(stack) < 1:
                feat += [(2, EMP)]
                feat += [(3, EMP)]
            else:
                feat += [(2, words[stack[-1]])]
                feat += [(3, tags[stack[-1]])]
        else:
            feat += [(2, words[stack[-1]])]
            feat += [(3, tags[stack[-1]])]
            
            feat += [(4, words[stack[-2]])]
            feat += [(5, tags[stack[-2]])]
        
        return feat
    
    def finalize(self):
        """Averages the weight vectors."""
        self.classifier.finalize()
        self.tagger.finalize()

