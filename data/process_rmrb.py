import random
random.seed(2020)

def read_file(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = []
        for line in f:
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


if __name__ == "__main__":
    sen_file = "人民日报2014NER数据/source_BIO_2014_cropus.txt"
    label_file = "人民日报2014NER数据/target_BIO_2014_cropus.txt"
    sens = read_file(sen_file)
    labels = read_file(label_file)
    # shuffle
    data = list(zip(sens, labels))
    random.shuffle(data)
    sens, labels = zip(*data)

    dev_length = int(len(sens)*0.1)
    write_file(sens[:dev_length], labels[:dev_length], "dev.txt")
    write_file(sens[dev_length+1:2*dev_length+1], labels[dev_length+1:2*dev_length+1], "test.txt")
    write_file(sens[2*dev_length+1:], labels[2*dev_length+1:], "train.txt")
