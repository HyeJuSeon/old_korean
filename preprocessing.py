# -*- coding: utf-8 -*-
import os
import json
from pprint import pprint
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import glob
from scipy.ndimage.filters import median_filter

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/MALGUNSL.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

base_dir = 'D:/old_korean'
label_dir = f'{base_dir}/labeling_data'
img_dir = f'{base_dir}/image_data'
'''
image_data/목판본_001_문학류_춘향전_01/목판본_001_문학류_춘향전_01_010.png
labeling_data/목판본_001_문학류_춘향전_01/목판본_001_문학류_춘향전_01_010.json
    'Image_filename': '목판본_001_문학류_춘향전_01_010',
    'Text_Coord': [{'annotate': '비', 'bbox': [705, 208, 48, 50, 0, 0]},
'''

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        img_array = np.fromfile(filename, dtype)
        return cv2.imdecode(img_array, flags)
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def dump_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def annotations(file):
    result = []
    with open(file, encoding='utf-8-sig') as f:
        annotation = json.load(f)
    for annotation in annotation['Text_Coord']:
        bbox = annotation['bbox'][:4]
        text = annotation['annotate']
        result.append((bbox, text))
    return result

def eda(phase):
    dirs = os.listdir(f'{img_dir}/{phase}')
    num_wood = 0
    num_transcript = 0
    num_typing = 0
    for dir in tqdm(dirs):
        path = f'{img_dir}/{phase}/{dir}'
        if '목판본' in dir:
            num_wood += len(os.listdir(path))
        elif '필사본' in dir:
            num_transcript += len(os.listdir(path))
        else:
            num_typing += len(os.listdir(path))
    print(num_wood, num_transcript, num_typing, num_wood + num_transcript + num_typing)

    x_ticks = ['필사본', '목판본', '활자본']
    idx = np.arange(len(x_ticks))
    plt.bar(idx, [num_transcript, num_wood, num_typing])
    plt.xticks(idx, x_ticks)
    plt.savefig(f'eda_{phase}.png')
    plt.show()

def preprocessing(phase):
    labels = open(f'{base_dir}/data/gt_{phase}.txt', 'w', encoding='utf-8')
    texts = set()
    dirs = os.listdir(f'{label_dir}/{phase}')
    # files = [_ for _ in os.listdir(f'{base_dir}/{phase}/[label]{phase}/1.간판') if _.endswith(file_ext)]
    # print(len(files))
    # print(len(os.listdir(f'{base_dir}/{phase}/[source]{phase}_간판_가로형간판')))
    for dir in tqdm(dirs):
        files = os.listdir(f'{label_dir}/{phase}/{dir}')
        for file in tqdm(files):
            annotation = annotations(f'{label_dir}/{phase}/{dir}/{file}')
            for i, anno in enumerate(annotation):
                bbox, text = anno
                x, y, width, height = bbox
                img = imread(f'{img_dir}/{phase}/{dir}/{file[:-5]}.png')
                try:
                    cropped = img[y:y + height, x:x + width]
                except:
                    print(f'{img_dir}/{phase}/{dir}/{file[:-5]}.png')
                    break
                if os.path.isfile(f'{base_dir}/data/{phase}/{file[:-5]}_{i}.png'):
                    labels.write(f'{phase}/{file[:-5]}_{i}.png\t{text}\n')
                    continue
                imwrite(f'{base_dir}/data/{phase}/{file[:-5]}_{i}.png', cropped)
                # texts.add(text)
                labels.write(f'{phase}/{file[:-5]}_{i}.png\t{text}\n')
    labels.close()
    # dump_pickle('texts.pkl', texts)

def concatenate():
    read_files = glob.glob(f'{base_dir}/*.txt')
    with open('gt_train.txt', 'w', encoding='utf-8') as outfile:
        for file in tqdm(read_files):
            with open(file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

def get_labels():
    labels = set()
    files = ['gt_train.txt', 'gt_val.txt']
    for file in files:
        for line in open(file, 'r', encoding='utf-8'):
            _, label = line.strip().split('\t')
            labels.add(label)
    korean = '가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘'
    for char in korean:
        labels.add(char)
    dump_pickle('labels.pkl', labels)
    print(labels, len(labels))
    with open('labels.txt', 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label)

def unsharp(phase, sigma, strength):
    path_origin = f'{base_dir}/data/{phase}'
    path_unsharp = f'{base_dir}/unsharp_data/{phase}'
    files = os.listdir(path_origin)
    for file in tqdm(files):
        if os.path.isfile(f'{path_unsharp}/{file}'):
            continue
        img = imread(f'{path_origin}/{file}', cv2.IMREAD_GRAYSCALE)
        image_mf = median_filter(img, sigma)
        lap = cv2.Laplacian(image_mf, cv2.CV_64F)
        sharp = img - strength * lap
        sharp[sharp > 255] = 255
        sharp[sharp < 0] = 0
        imwrite(f'{path_unsharp}/{file}', sharp)

if __name__ == '__main__':
    # bbox 확인
    winname = 'result'
    img = imread('목판본_001_문학류_춘향전_01_010.png')
    x, y = 705, 208
    cv2.circle(img, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    x, y = 705 + 48, 208 + 50
    cv2.circle(img, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
    cv2.moveWindow(winname, 40, 30)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindow()

    eda('train')
    eda('val')
    preprocessing('train')
    preprocessing('val')
    concatenate()
    get_labels()
    unsharp('train', 5, 0.8)
    unsharp('val', 5, 0.8)
    unsharp('test', 5, 0.8)