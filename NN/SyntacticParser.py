def train(path, end=200, epochs=1):
	import CONLLUIO
	import Extended_Parser as Parser

	par = Parser.Parser()
	for k in range(epochs):
		count = 0
		for gold in CONLLUIO.read_file(path):
			print("\rtrain #{:5d} {:5d}".format(k+1,count), end="")
			par.update(*gold)
			count += 1
			if count > end:
				break

	par.finalize()
	return par

def print_tree(parser, test_file, out):
    import CONLLUIO
    import Extended_Parser as Parser
    import codecs
    import sys
    import io
    import os
    import fileinput
    import datetime

    date_time = str(datetime.datetime.today().replace(microsecond=0))#.rsplit(sep=":",maxsplit=-2)
    THISDIR = os.path.dirname(os.path.abspath(__file__))
    scores = codecs.open(THISDIR + "/output/scores-" + out + "-" + "".join(date_time) + ".txt","w","utf-8")
    #out = codecs.open(THISDIR + "/output/" + out + "-" + "".join(date_time) + ".txt","w","utf-8")

    acc_k = acc_n = 0
    uas_k = uas_n = 0

    for i, (words, gold_tags, gold_tree) in enumerate(CONLLUIO.read_file(test_file)):
        if i < 100:
            pred_tags, pred_tree = parser.parse(words)
            #print(u",".join(pred_tags) + "\n" + u"".join(str(pred_tree)) + "\n",file=out)
            acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
            acc_n += len(words) - 1
            uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
            uas_n += len(words) - 1
            print("\rParsing sentence #{}".format(i), end="")
    print("")
    print("Tagging accuracy: {:.2%}".format(acc_k / acc_n), file=scores)
    print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n), file=scores)
