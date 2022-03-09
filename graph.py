import re
import matplotlib.pyplot as plt

def extract_num_loss(string):
    str1, str2, str3, str4 = re.findall(r'\d+', string[13:])[:-2]
    return float(f'{str1}.{str2}'), float(f'{str3}.{str4}')

def extract_num_acc(string):
    str1, str2, _, _ = re.findall(r'\d+', string)
    return float(f'{str1}.{str2}')

def graph(dataset='origin'):
    '''
    loss
    [1/300000] Train loss: 78.00769, Valid loss: 74.62008, Elapsed_time: 1.55412\n

    acc
    Current_accuracy : 0.000, Current_norm_ED  : 0.00\n
    Best_accuracy    : 0.000, Best_norm_ED     : 0.00\n
    '''
    with open(f'log_train_{dataset}.txt', 'r', encoding='utf-8') as f:
        file = f.readlines()
    loss_log = [line for line in file if 'loss' in line]
    acc_log = [line for line in file if 'accuracy' in line and 'Current' in line]
    loss = list(map(extract_num_loss, loss_log))
    acc = list(map(extract_num_acc, acc_log))

    train_loss = [t_loss for t_loss, _ in loss]
    val_loss = [v_loss for _, v_loss in loss]
    epochs_range = range(0, len(loss) * 100, 100)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs_range, acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Validation Accuracy')
    plt.savefig(f'img/{dataset}_data_result.png')
    # plt.show()


if __name__ == '__main__':
    graph()
