import argparse
import gzip
import json
import re
import string
import spacy
import en_core_web_sm
# from utils import load_cached_embeddings, Indexer, WordEmbeddings
from utils_qc import Indexer, WordEmbeddings, read_word_embeddings
import ne_model
from ne_model import *

def read_data(gold_file):
    """Reads answers from dataset file. Each question (marked by its qid)
    can have multiple possible answer spans.

    Args:
        gold_file: Path to dataset file (string).

    Returns:
        True dict mapping question id (id) to answer span(s).
    """
    questions = {}
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']
                questions[qa['qid']] = qa['question']
    return questions, answers


    # mostly look like sentiment_classifier.py form A2, but we have to coonvert the data we're working with to be similar to what is used there
    # we will also have to make sure that there is no dependency on yes/no or 0/1 results (predict is a great example) 
    # MAKE SURE TO USE torch.save()
    # relevant pieces in *fp* code are tagged with breakpoints


def read_sentiment_examples(questions, answers) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    NOTE: Compared to Assignment 1, we lowercase the data for you. This is because the GloVe embeddings don't
    distinguish case and so can only be used with lowercasing.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """

    exs = []
    label = 0 #LABELS SHOULD HAVE A SEPARATE INDEXER
    ne_indexer = Indexer()
    
    ne_indexer.add_and_get_index("N/A")
    ne_indexer.add_and_get_index("PERSON")
    ne_indexer.add_and_get_index("NORP")
    ne_indexer.add_and_get_index("FAC")
    ne_indexer.add_and_get_index("ORG")
    ne_indexer.add_and_get_index("GPE")
    ne_indexer.add_and_get_index("LOC")
    ne_indexer.add_and_get_index("PRODUCT")
    ne_indexer.add_and_get_index("EVENT")
    ne_indexer.add_and_get_index("WORK_OF_ART")
    ne_indexer.add_and_get_index("LAW")
    ne_indexer.add_and_get_index("LANGUAGE")
    ne_indexer.add_and_get_index("DATE")
    ne_indexer.add_and_get_index("TIME")
    ne_indexer.add_and_get_index("PERCENT")
    ne_indexer.add_and_get_index("MONEY")
    ne_indexer.add_and_get_index("QUANTITY")
    ne_indexer.add_and_get_index("ORDINAL")
    ne_indexer.add_and_get_index("CARDINAL")

    nlp = en_core_web_sm.load()
    counter=0
    for q_key in questions:
        tokenized_cleaned_sent = list(filter(lambda x: x != '', questions[q_key].lower().rstrip().split(" ")))
        # here we determine label using spacy
        label=0
        counter+=1
        if(counter%5000==0):
            print("break")
            break
        for answer in range(len(answers[q_key])):
            if(label == 0):
                a_token = nlp(answers[q_key][answer])
                if len(a_token.ents)==1:
                    label = ne_indexer.index_of(a_token.ents[0].label_)
        exs.append(SentimentExample(tokenized_cleaned_sent, label))
    return exs

def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    return print_evaluation([ex.label for ex in exs], classifier.predict_all([ex.words for ex in exs]))


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold labels
    :param predictions: pred labels
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        # if prediction == 1:
        #     num_pred += 1
        # if gold == 1:
        #     num_gold += 1
        # if prediction == 1 and gold == 1:
        #     num_pos_correct += 1
        num_total += 1
    acc = float(num_correct) / num_total
    output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
    # prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    # rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    # f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    # output_str += "; Precision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
    # output_str += "; Recall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
    # output_str += "; F1 (harmonic mean of precision and recall): %f" % f1
    print(output_str)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='path to training dataset')
    parser.add_argument('--dev_path', type=str, help='path to dev dataset')
    parser.add_argument('--word_vecs_path', type=str, help="path to glove word embeddings")
    args = parser.parse_args()

    # Load train, dev, and test exs and index the words.
    train_qs, train_as = read_data(args.train_path)
    dev_qs, dev_as = read_data(args.dev_path)
    train_exs = read_sentiment_examples(train_qs,train_as)
    dev_exs = read_sentiment_examples(dev_qs,dev_as)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " train/dev examples")


    # Returns Dictionary mapping words (strings) to vectors (list of floats).
    # Modify model to expect a dictionary instead of what it has rn. (init function in DANN class in A2 models)
    word_embeddings = read_word_embeddings(args.word_vecs_path) 

    # Train and evaluate
    model = train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings)
    print("=====Train Accuracy=====")
    train_acc = evaluate(model, train_exs)
    print("=====Dev Accuracy=====")
    dev_acc = evaluate(model, dev_exs)


    data = {'dev_acc': dev_acc}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    # with open("../results/results.json", 'w') as outfile:
    #     json.dump(data, outfile)