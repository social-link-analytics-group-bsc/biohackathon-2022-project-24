import csv
import argparse
import json
from nltk.tokenize import word_tokenize

def parsing_arguments(parser):
    parser.add_argument("--data", type=str, default='annotation/trial.jsonl',
                        help='Prodigy data to transform into conll')
    return parser

def extract_labels(line):
    token_with_label = []
    token_to_label = {}
    try:
        for span in line['spans']:
            if span['token_start'] == span['token_end']:
                token_with_label.append(line['tokens'][span['token_start']]['id'])
                token_to_label[line['tokens'][span['token_start']]['id']] = span['label']
            else:
                for r in range(span['token_start'], span['token_end']+1): # TODO: finish this
                    token_with_label.append(line['tokens'][r]['id'])
                    token_to_label[line['tokens'][r]['id']] = span['label']
    except KeyError: # if there are no spans
        return [], {}
    return token_with_label, token_to_label

def write_conll(split, data, max=None):
    counts = 0
    with open(split, 'w') as o:
        writer = csv.writer(o, delimiter='\t')
        for i,line in enumerate(data):
            counts += 1
            token_with_label, token_to_label = extract_labels(line)
            for token in line['tokens']:
                if token['id'] in token_with_label:
                    writer.writerow([token['text'], token_to_label[token['id']]])
                else:
                    writer.writerow([token['text'], 'O'])
            writer.writerow([])
            if counts == max:
                print(split)
                print(i)
                return data[i:]
        return counts

def main():
    parser = argparse.ArgumentParser()
    parser = parsing_arguments(parser)
    args = parser.parse_args()
    print(args)

    with open(args.data , 'r') as file:
        f = file.readlines()
        data = []
        for line in f:
            data.append(json.loads(line))

    print(len(data))
    print(data[0])

    remaining_data = write_conll('training_data/train.conll', data, 800)
    print(len(remaining_data))
    remaining_data = write_conll('training_data/dev.conll', remaining_data, 100)
    print(len(remaining_data))
    remaining_data = write_conll('training_data/test.conll', remaining_data, 100)
    print(len(remaining_data))
    left = write_conll('training_data/remaining.conll', remaining_data)
    print('Not used', left)
    print('Total', 300+50+50+left)





if __name__ == "__main__":
    main()