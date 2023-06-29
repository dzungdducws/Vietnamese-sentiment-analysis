import numpy as np
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8')

cvs = [1, 2, 3, 4, 5]

for cv in cvs:
    word2id = {}
    id2word = {}
    index = 1
    maxlen = 0
    avglen = 0
    count100 = 0

    # Đường dẫn file
    train_pos_file = f"data/data_token/fold_{cv}/train_nhan_1.txt"
    train_neg_file = f"data/data_token/fold_{cv}/train_nhan_0.txt"
    train_neu_file = f"data/data_token/fold_{cv}/train_nhan_2.txt"

    test_pos_file = f"data/data_token/fold_{cv}/test_nhan_1.txt"
    test_neg_file = f"data/data_token/fold_{cv}/test_nhan_0.txt"
    test_neu_file = f"data/data_token/fold_{cv}/test_nhan_2.txt"

    open_files = [train_pos_file, train_neg_file, train_neu_file,
                  test_pos_file, test_neg_file, test_neu_file]

    # Đường dẫn lưu file
    train_pos_save = f"data/data_token/fold_{cv}/train_pos"
    train_neg_save = f"data/data_token/fold_{cv}/train_neg"
    train_neu_save = f"data/data_token/fold_{cv}/train_neu"

    test_pos_save = f"data/data_token/fold_{cv}/test_pos"
    test_neg_save = f"data/data_token/fold_{cv}/test_neg"
    test_neu_save = f"data/data_token/fold_{cv}/test_neu"

    save_files = [train_pos_save, train_neg_save, train_neu_save,
                  test_pos_save, test_neg_save, test_neu_save]

    for open_file, save_file in zip(open_files, save_files):
        pos = []
        print(open_file)
        with open(open_file, 'r', encoding="utf8") as file:
            for aline in file.readlines():
                aline = aline.replace('\n', "")
                ids = np.array([], dtype="int32")
                ids = np.asanyarray(ids)
                for word in aline.split(' '):
                    word = word.lower()
                    if word in word2id:
                        ids = np.append(ids, word2id[word])
                    else:
                        if word != '':
                            word2id[word] = index
                            id2word[index] = word
                            ids = np.append(ids, index)
                            index += 1
                if len(ids) > 0:
                    pos.append(ids)
        # print(pos[8])
        print(len(pos))
        np.save(save_file, pos)
        for li in pos:
            # print(li)
            if maxlen < len(li):
                maxlen = len(li)
            avglen += len(li)
            if len(li) > 250:
                count100 += 1

    print(f"Fold: {cv}")
    # print(word2id) In ra các từ
    print(f"Tổng số từ trong từ điển chứa từ vựng : {len(word2id)}")
    print(f"Chiều dài tối đa của các câu: {maxlen}")
    print(f"Tổng chiều dài các câu: {avglen}")
    print(f"số câu có chiều dài lớn hơn 250: {count100}")
