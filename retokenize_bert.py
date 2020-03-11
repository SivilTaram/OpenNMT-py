from onmt.utils.bert_tokenization import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer(vocab_file="bert-base-chinese/vocab.txt")


def retokenize_sentence(utterance):
    tokens = tokenizer.tokenize(utterance)
    return " ".join(tokens)


def retokenize_file(read_file_path, write_file_path):
    with open(read_file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines):
            # some english tokens will be affected by BERT-based tokenizing (chinese character is not affected)
            new_utt = retokenize_sentence(line.strip('\n'))
            new_lines.append(new_utt)

    with open(write_file_path, "w", encoding="utf8") as f:
        f.write("\n".join(new_lines))


if __name__ == '__main__':
    retokenize_file("data/dial-src-train.txt", "data/dial-src-train-bert.txt")
    retokenize_file("data/dial-src-val.txt", "data/dial-src-val-bert.txt")
    retokenize_file("data/dial-tgt-train.txt", "data/dial-tgt-train-bert.txt")
    retokenize_file("data/dial-tgt-val.txt", "data/dial-tgt-val-bert.txt")
