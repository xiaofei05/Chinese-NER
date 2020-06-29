import random
import json
random.seed(2020)

def read_file(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = []
        for line in f:
            if "target" in filename:
                line = line.replace("_", "-")
            data.append(line.strip().split())
    return data

def write_file(sens, labels, file_name):
    assert len(sens)==len(labels)
    with open(file_name, "w", encoding="utf8") as f:
        for i in range(len(sens)):
            assert len(sens[i])==len(labels[i])
            for j in range(len(sens[i])):
                f.write(sens[i][j]+"\t"+labels[i][j]+"\n")
            f.write("\n")
    print(file_name + "'s datasize is " , len(sens))


def get_dict(sents, filter_word_num):
    word_count = {}
    for sent in sents:
        for word in sent:
            word_count[word] = word_count.get(word, 0) + 1
    
    # 过滤低频词
    word2id = {
        "[PAD]": 0, 
        "[UNK]": 1
    }
    for word, count in word_count.items():
        if count >= filter_word_num:
            word2id[word] = len(word2id)
    
    print("Total %d tokens, filter count<%d tokens, save %d tokens."%(len(word_count)+2, filter_word_num, len(word2id)))

    return word2id, word_count

if __name__ == "__main__":
    sen_file = "data/renminribao2014/source_BIO_2014_cropus.txt"
    label_file = "data/renminribao2014/target_BIO_2014_cropus.txt"
    sens = read_file(sen_file)
    labels = read_file(label_file)
    # get dicts
    word2id, _ = get_dict(sens, filter_word_num=5)
    with open("data/word2id.json", "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False)
    # shuffle
    data = list(zip(sens, labels))
    random.shuffle(data)
    sens, labels = zip(*data)

    dev_length = int(len(sens)*0.1)

    # write_file(sens[:1000], labels[:1000], "data/dev.txt")
    # write_file(sens[1000:2000], labels[1000:2000], "data/test.txt")
    # write_file(sens[10000:30000], labels[10000:30000], "data/train.txt")

    write_file(sens[:dev_length], labels[:dev_length], "data/dev.txt")
    write_file(sens[dev_length:2*dev_length], labels[dev_length:2*dev_length], "data/test.txt")
    write_file(sens[2*dev_length:], labels[2*dev_length:], "data/train.txt")
    