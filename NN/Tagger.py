from collections import defaultdict

class Perceptron(object):

    def __init__(self):
        self.labels = set()
        self.w = defaultdict(lambda : defaultdict(float))
        self.cnt = 1
        self.acc = defaultdict(lambda : defaultdict(float))

    def predict(self, x, candidates=None):
        if candidates is None:
            candidates = self.labels
        scores = {c:0.0 for c in candidates}
        for f in x:
            if f in self.w:
                for c, w in self.w[f].items():
                    if c in candidates:
                        scores[c] += w

        return max(scores, key=lambda c: (scores[c], c))

    def update(self, x, y):
        self.labels.add(y)
        pred = self.predict(x)
        if pred != y:
            for i in x:
                self.w[i][y] += 1
                self.w[i][pred] -= 1
                self.acc[i][y] += self.cnt
                self.acc[i][pred] -= self.cnt
        self.cnt += 1
        return pred

    def finalize(self):
        for i in self.w:
            for c in self.w[i]:
                self.w[i][c] -= self.acc[i][c] / self.cnt

    @classmethod
    def train(cls, data, n_epochs=1,avg=True):
        ret = cls()
        for e in range(n_epochs):
            for x, y in data:
                ret.update(x, y)
        if avg==True:
            ret.finalize()
        return ret

class Tagger(object):
    def __init__(self):
        self.classifier = Perceptron()
        self.feature_index = 0
        self.feature_map = {}
        self.n=0
        
    def features(self, words, i, pred_tags):
        def feature2index(feature):
            if feature not in self.feature_map:
                self.feature_map[feature] = self.feature_index
                self.feature_index += 1
            return self.feature_map[feature]
            
        def make_features(words, i, pred_tags):
            features=[]
            
            #feature window[i-1,i,i+1] and pred_tags[i-1]
            window_words=['<BOS>']
            window_words.extend(words)
            window_words+=['<EOS>']
                    
            window_ptags =['<TAG_BOS>']
            window_ptags.extend(pred_tags)
            
            i=i+1
                        
            #one word
            features+=[(len(features),window_words[i - 1])]
            features+=[(len(features),window_words[i + 0])]*3
            features+=[(len(features),window_words[i + 1])]
            
            #two words
            features+=[(len(features),window_words[i - 1]+"<2w>"+window_words[i + 0])]
            features+=[(len(features),window_words[i + 0]+"<2w>"+window_words[i + 1])]
            features+=[(len(features),window_words[i - 1]+"<2w>"+window_words[i + 1])]
            
            #three words
            features+=[(len(features),window_words[i - 1]+"<3w>"+window_words[i + 0]+"<3w>"+window_words[i + 1])]
            
            #len of the word
            features+=[(len(features),"<len>"+str(len(window_words[i])))]
            features+=[(len(features),"<len>"+str(len(window_words[i-1])))]
            features+=[(len(features),"<len>"+str(len(window_words[i+1])))]
            
            #one ptage
            features+=[(len(features),window_ptags[i - 1])]
            
            #one ptage and one word
            features+=[(len(features),window_ptags[i - 1]+"<1p1w>"+window_words[i - 1])]
            features+=[(len(features),window_ptags[i - 1]+"<1p1w>"+window_words[i + 0])]
            features+=[(len(features),window_ptags[i - 1]+"<1p1w>"+window_words[i + 1])]
            
            #one ptage and two words
            features+=[(len(features),window_ptags[i - 1]+"<1p2w>"+window_words[i - 1]+"<1p2w>"+window_words[i + 0])]
            features+=[(len(features),window_ptags[i - 1]+"<1p2w>"+window_words[i + 0]+"<1p2w>"+window_words[i + 1])]
            features+=[(len(features),window_ptags[i - 1]+"<1p2w>"+window_words[i - 1]+"<1p2w>"+window_words[i + 1])]
            
            #one ptage and three words
            features+=[(len(features),window_ptags[i - 1]+"<1p3w>"+window_words[i - 1]+"<1p3w>"+window_words[i + 0]+"<1p3w>"+window_words[i + 1])]
            
            #suffix
            features+=[(len(features),"<suffix>"+window_words[i][-1:])]
            features+=[(len(features),"<suffix>"+window_words[i][-2:])]
            features+=[(len(features),"<suffix>"+window_words[i][-3:])]
            features+=[(len(features),"<suffix>"+window_words[i][-4:])]
            features+=[(len(features),"<suffix>"+window_words[i][-5:])]
            
            #prefix
            features+=[(len(features),"<prefix>"+window_words[i][:1])]
            features+=[(len(features),"<prefix>"+window_words[i][:2])]
            features+=[(len(features),"<prefix>"+window_words[i][:4])]
            features+=[(len(features),"<prefix>"+window_words[i][:5])]
            
            #suffix1
            features+=[(len(features),"<suffix1>"+window_words[i+1][-2:])]
            features+=[(len(features),"<suffix1>"+window_words[i+1][-4:])]
            features+=[(len(features),"<suffix1>"+window_words[i+1][-5:])]
            
            #prefix1
            features+=[(len(features),"<prefix1>"+window_words[i+1][:2])]
            features+=[(len(features),"<prefix1>"+window_words[i+1][:4])]
            
            return features
            
        ret=[]
        for feature in make_features(words, i, pred_tags):
            ret.append(feature)
        return ret
        
    def tag(self, words):
        pred_tags = []
        for i in range(len(words)):
            features = self.features(words, i, pred_tags)
            pred_tags.append(self.classifier.predict(features))
        return pred_tags

    def update(self, words, gold_tags):
        pred_tags = []
        for i, g in enumerate(gold_tags):
            features = self.features(words, i, pred_tags)
            pred_tags.append(self.classifier.update(features, g))
        return pred_tags

    def train(self, data,n_epochs=1):
        for sentence_with_tags in data*n_epochs:
            words, gold_tags = zip(*sentence_with_tags)
            self.update(words, gold_tags)
        self.classifier.finalize()

    def finalize(self):
        self.classifier.finalize()

