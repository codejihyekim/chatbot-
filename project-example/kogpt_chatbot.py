import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from emo import Emotion
from bertClassifier import BERTClassifier


class Solution:

    def __init__(self) -> None:
        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.BOS = '</s>'
        self.EOS = '</s>'
        self.MASK = '<unused0>'
        self.SENT = '<unused1>'
        self.PAD = '<pad>'

    def chatbot(self):
        Q_TKN = self.Q_TKN
        SENT = self.SENT
        A_TKN = self.A_TKN
        BOS = self.BOS
        EOS = self.BOS
        PAD = self.PAD 
        MASK = self.MASK
        device = torch.device("cpu")
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK)
        model = torch.load('chatbot-gpt.pt', map_location=device)
        with torch.no_grad():
            while 1:
                q = input("나 > ").strip()
                if q == "quit":
                    break
                elif q == "#":
                    Emotion.test(self)
                    continue
                a = ""
                while 1:
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                    pred = model(input_ids)
                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("mibot > {}".format(a.strip()))

if __name__ == '__main__':
    Solution().chatbot()