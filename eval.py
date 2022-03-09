def replace(str):
    str = str[:-7].strip()
    return str.replace('목판본_001_문학류_춘향전', '목판본_001_문학류_춘향전')

def pred_file_preprocessing(dataset):
    with open(f'log_eval_result_{dataset}.txt', 'r', encoding='utf-8') as f:
        file = f.readlines()
    file = [line for line in file if 'test' in line]
    file = list(map(replace, file))
    return file

def test_file():
    with open('gt_train.txt', 'r', encoding='utf-8') as f:
        file = f.readlines()
    with open('gt_test.txt', 'w', encoding='utf-8') as f:
        for line in file:
            if '목판본_001_문학류_춘향전' in line:
                f.write(line)

def eval_log(dataset='origin'):
    '''
    label: train/목판본_001_문학류_춘향전_01_011_73.png	류
    pred: /content/gdrive/MyDrive/old_korean_ocr/test/목판본_001_문학류_춘향전_01_011_73.png	류                        	0.3353
    '''
    with open('gt_test.txt', 'r', encoding='utf-8') as f:
         label_file = f.readlines()
    pred_file = pred_file_preprocessing(dataset)
    label_dict = dict()
    pred_dict = dict()
    imgs = set()
    for label_line, pred_line in zip(label_file, pred_file):
        file, label = label_line.split('\t')
        idx = file.find('목')
        label_dict[file[idx:]] = label.strip()
        imgs.add(file[idx:])
        try:
            file, pred = pred_line.strip().split('\t')
        except:
            file = pred_line.strip()
            pred = ' '
        idx = file.find('목')
        pred_dict[file[idx:]] = pred

    correct = 0
    for file in list(imgs):
        try:
            if pred_dict[file] == label_dict[file]:
                correct += 1
        except:
            continue
    print(f'Test accuracy: {correct / len(imgs):0.4f}')

if __name__ == '__main__':
    eval_log()
