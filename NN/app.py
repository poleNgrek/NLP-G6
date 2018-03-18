import SyntacticParser as sp
import os
import sys

if __name__ == "__main__":


    #english_train = sys.argv[1]
    #english_test = sys.argv[2]
    #swedish_train = sys.argv[3]
    #swedish_test = sys.argv[4]
    #training_samples = int(sys.argv[5])
    if len(sys.argv) == 6:
        epochs = int(sys.argv[5])
    else:
        epochs = 1

    english_train = '../data/en-ud-train-projective.conllu'
    english_test = '../data/en-ud-test.conllu'
    swedish_train = '../data/sv_lines-ud-train.conllu'
    swedish_test = '../data/sv_lines-ud-test.conllu'
    training_samples = 250
    epochs = 1

    prj_en_train = '../data/en-ud-projectivized-train.conllu'
    prj_en_test = '../data/en-ud-projectivized-test.conllu'
    prj_sv_train = '../data/sv-ud-projectivized-train.conllu'
    prj_sv_test = '../data/sv-ud-projectivized-test.conllu'
    os.system('python3 projectivize.py ' + '<' + english_train + '>' + ' ' + prj_en_train)
    os.system('python3 projectivize.py ' + '<' + english_test + '>' + ' ' + prj_en_test)
    os.system('python3 projectivize.py ' + '<' + swedish_train + '>' + ' ' + prj_sv_train)
    os.system('python3 projectivize.py ' + '<' + swedish_test + '>' + ' ' + prj_sv_test)

    en_par = sp.train(prj_en_train, training_samples, epochs)
    sp.print_tree(en_par, prj_en_test, 'en_output')

    sv_par = sp.train(prj_sv_train, training_samples, epochs)
    sp.print_tree(sv_par, prj_sv_test, 'sv_output')