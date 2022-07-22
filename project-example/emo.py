
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import random
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers.optimization import get_cosine_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from bertClassifier import BERTClassifier
from bertDataset import BERTDataset


class Emotion:
    def __init__(self):
        self.max_len = 100
        self.batch_size = 16
        bertmodel, vocab = get_pytorch_kobert_model()
        #토큰화
        tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    def test(self):
        max_len = 100
        batch_size = 16
        bertmodel, vocab = get_pytorch_kobert_model()
        #토큰화
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        device = torch.device("cuda:0")

        with torch.no_grad():
            end = 1
            while end == 1 :
                sentence = input("감정관련 이야기를 들려주세요 \n")
                #self.music(sentence)
                model = torch.load('emotion.pt')
                data = [sentence, '0']
                dataset_another = [data]

                another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
                test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

                model.eval()

                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                    token_ids = token_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)

                    valid_length= valid_length
                    label = label.long().to(device)

                    out = model(token_ids, valid_length, segment_ids)
                    
                    sad_music = ['사랑안해', '그때 그순간 그대로', '보고싶었어', '바보가 바보에게', '다정히 내 이름을 부르면', '드라마']
                    happy_music = ['여름여름해', '빨간 맛', '아주 NICE', 'PARTY', '마지막처럼']
                    angry_music = ['그건 니 생각이고', '대취타', '팩트폭행', '작두']
                    a = {'a':1, 'b':2, }

                    test_eval=[]
                    for i in out:
                        logits=i
                        logits = logits.detach().cpu().numpy()

                        if np.argmax(logits) == 0:
                            test_eval.append("당황스러우셨군요 ")
                        elif np.argmax(logits) == 1:
                            test_eval.append("화가나셨군요 ")
                        elif np.argmax(logits) == 2:
                            test_eval.append("불안하시군요 ")
                        elif np.argmax(logits) == 3:
                            test_eval.append("행복하시군요 ")
                        elif np.argmax(logits) == 4:
                            test_eval.append("슬프시군요 ")

                    if test_eval[0] == "행복하시군요 ":
                        print(">> " + test_eval[0] + random.choice(happy_music) + "을/를 추천해드려요")
                    elif test_eval[0] == "슬프시군요 ":
                        print(">> " + test_eval[0] + random.choice(sad_music) + "을/를 추천해드려요")
                break
                '''if sentence == '그럼 이만' :
                    break
                print("\n")'''

if __name__ == '__main__':
    Emotion().test()