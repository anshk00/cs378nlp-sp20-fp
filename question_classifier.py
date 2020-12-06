def read_answers(gold_file):
    """Reads answers from dataset file. Each question (marked by its qid)
    can have multiple possible answer spans.

    Args:
        gold_file: Path to dataset file (string).

    Returns:
        True dict mapping question id (id) to answer span(s).
    """
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']
                print("Questions: ", qa["question"])
    return answers

#Either read data like this ^ or use Dataset to parse/tokenize from data.py

def main(args):
    print("Is main")

    # mostly look like sentiment_classifier.py form A2, but we have to coonvert the data we're working with to be similar to what is used there
    # we will also have to make sure that there is no dependency on yes/no or 0/1 results (predict is a great example) 
    # MAKE SURE TO USE torch.save()
    # relevant pieces in *fp* code are tagged with breakpoints


def data_converter():
    # first: do we take from data.py extraction (tokenized but incorrectly formatted data) or from read_answers in evaluate (correct data but non-tokenized)
    # next, convert that to *distributed* sentiment examples (questions are just tokenized, answers are converted to a "one-hot" index)
    # from here it should be similar enough to use the A2 as a basis



if __name__ == '__main__':
    main(parser.parse_args())