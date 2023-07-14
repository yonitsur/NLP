import utils

if __name__ == "__main__":
    filepath = "birth_dev.tsv"
    with open(filepath, encoding='utf-8') as fin:
        pred = ["London" for _ in fin.readlines()]
    total, correct = utils.evaluate_places(filepath, pred)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('no targets provided')
                    

