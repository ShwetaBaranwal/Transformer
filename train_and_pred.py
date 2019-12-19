# -*- coding: utf-8 -*-

from config_file import config_set
from __future__ import unicode_literals
import numpy as np
from transformer import get_model, decode

#calling configuration class
hp = config_set()

#downloading dataset
!bash download.sh

#preprocessing dataset
!python data_load.py

#importing sentencepiece tokenizor model
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(hp.embed_model)


class final_data():
    """
    reads input and output document and modifies it in the form that model takes
    """

    def __init__(self, vocab_fpath, fpath1, fpath2, maxlen1, maxlen2):
      """
      :param vocab_fpath: vocabulary file path
      :param fpath1: source document (German)
      :param fpath2: target document (English)
      :param maxlen1: max sentence length of german document
      :param maxlen2: max sentence length of english document
      Returns
      Intialization
      """
      self.vocab_fpath = vocab_fpath
      self.fpath1 = fpath1
      self.fpath2 = fpath2
      self.maxlen1 = maxlen1
      self.maxlen2 = maxlen2


    def load_vocab(self, vocab_path):
      """
      Loads vocabulary file and returns idx<->token maps
      :param vocab_path: (string) vocabulary file path
      Note that these are reserved
      0: <pad>, 1: <unk>, 2: <s>, 3: </s>

      sample of vocab file:
      <pad>	0
      <unk>	0
      <s>	0
      </s>	0
      en	-0
      er	-1
      in	-2
      ▁t	-3
      ch	-4
      ▁a	-5
      ▁d	-6
      ▁w	-7
      ▁s	-8
      ▁th	-9
      nd	-10
      ie	-11

      Returns
      two dictionaries.
      """
      vocab = [line.split()[0] for line in open(vocab_path, 'r').read().splitlines()]
      token2idx = {token: idx for idx, token in enumerate(vocab)}
      idx2token = {idx: token for idx, token in enumerate(vocab)}
      return token2idx, idx2token


    def load_data(self, path1, path2, T1, T2):
      """
      Loads source and target data and filters out too lengthy samples.
      :param path1: source file path. string.
      :param path2: target file path. string.
      :param T1: source sent maximum length. scalar.
      :param T2: target sent maximum length. scalar.
      Returns
      sents1: list of source sents
      sents2: list of target sents
      """
      sents1, sents2 = [], []
      with open(path1, 'r') as f1, open(path2, 'r') as f2:
          for sent1, sent2 in zip(f1, f2):
              if len(sent1.split()) + 1 > T1: continue # 1: </s>
              if len(sent2.split()) + 1 > T2: continue  # 1: </s>
              sents1.append(sent1.strip())
              sents2.append(sent2.strip())

      return sents1, sents2


    def encode(self, inp, type, dict, T1, T2):
      """
      Converts string to number.
      :param inp: input sentence tokens
      :param type: "x" (source side) or "y" (target side)
      :param dict: token2idx dictionary
      :param T1: source sent maximum length. scalar.
      :param T2: target sent maximum length. scalar.
      Returns
      list of numbers
      """
      inp_str = str(inp)
      if type=="x":
        tokens = inp_str.split() + ["</s>"] + ['<pad>'] * (T1 - len(inp_str.split()) - 1)
      else:
        tokens = ["<s>"] + inp_str.split() + ["</s>"] + ['<pad>'] * (T2 - len(inp_str.split()) - 1)

      x = [dict.get(t, dict["<unk>"]) for t in tokens]
      return x


    def padded_array(self, sents1, sents2):
      """
      Generates training / evaluation data
      sents1: list of source sents
      sents2: list of target sents
      Returns
      encoder input
      decoder input
      decoder output
      """
      encode_input=[]
      decode_input=[]
      decode_output_=[]
      self.token2idx, self.idx2token = self.load_vocab(self.vocab_fpath)
      for sent1, sent2 in zip(sents1, sents2):
          x = self.encode(sent1, "x", self.token2idx, self.maxlen1, self.maxlen2)
          y = self.encode(sent2, "y", self.token2idx, self.maxlen1, self.maxlen2)
          decoder_inp, y = y[:-1], y[1:]
          encode_input.append(x)
          decode_input.append(decoder_inp)
          decode_output_.append(y)

      decode_output = [list(map(lambda r: [r], r)) for r in decode_output_]

      print("total lines in encoder input = {}".format(len(encode_input)))
      print("total lines in decoder input = {}".format(len(decode_input)))
      print("total lines in decoder output = {}".format(len(decode_output)))

      return encode_input, decode_input, decode_output


    def compute(self):
      """
      calls all definition in order
      """
      xs, ys = self.load_data(self.fpath1, self.fpath2, self.maxlen1, self.maxlen2)
      x, y_in, y_out = self.padded_array(xs, ys)
      return x, y_in, y_out



# hp.maxlen1, hp.maxlen2 = 20,20
#calling training data
m = final_data(hp.vocab, hp.train1, hp.train2, hp.maxlen1, hp.maxlen2)
x, y_in, y_out = m.compute()

#getting model
model = get_model(
            token_num=len(m.token2idx),
            embed_dim=32,
            encoder_num=2,
            decoder_num=2,
            head_num=4,
            hidden_dim=128,
            dropout_rate=0.05,
            use_same_embed=True,  # Use different embeddings for different languages
        )

model.compile('adam', 'sparse_categorical_crossentropy')

model.summary()

model.fit(
            x=[np.array(x), np.array(y_in)],
            y=np.array(y_out),
            epochs=10,
            batch_size=32
        )

#calling evaluation/validation data
m = final_data(hp.vocab, hp.eval1, hp.eval2, hp.maxlen1, hp.maxlen2)
x_eval, y_in_eval, y_out_eval = m.compute()

#predicting
s_id, sentence, output_sentence = decode(
                                    model,
                                    x_eval,
                                    dict_token2idx = m.token2idx,
                                    dict_idx2token = m.idx2token,
                                    sp = sp
                                    )

#displaying few predicted output
for i in range(hp.sample_num_display):
    predicted = sentence[i]
    actual = ' '.join(map(lambda x: m.idx2token[x], y_in_eval[i]))
    print("predicted = {}".format(predicted))
    print("actual = {}".format(actual))
    print('\n')

#writing predicted output in file
with open(hp.predicted_output_file, "w") as fout:
    for sent in output_sentence:
        fout.write(sent + "\n")

#writing actual output in file
with open(hp.actual_output_file, "w") as fout:
    for sent in y_in_eval:
      s1 = list(map(lambda x: m.idx2token[x], sent))
      s2 = ' '.join(i.replace('▁','') for i in s1 if i not in ('<s>','</s>','<pad>'))
      fout.write(s2 + "\n")

#computing BLEU score for validation data
from nltk.translate.bleu_score import sentence_bleu
score = []
with open(hp.actual_output_file, 'r') as f1, open(hp.predicted_output_file, 'r') as f2:
    for actual, predict in zip(f1, f2):
        reference = actual.split()
        candidate = predict.split()
        score.append(sentence_bleu(reference, candidate))
print("Avg Bleu score for document is: {}".format(float(sum(score)/len(score))))
print("Min Bleu score for document is: {}".format(float(min(score))))
print("Max Bleu score for document is: {}".format(float(max(score))))

#plotting bleu score distribution
import random
from matplotlib import pyplot as plt

data = np.array(score)

# fixed bin size
bins = np.arange(0.0,1.0,0.1) # fixed bin size

plt.xlim([min(data)-.5, max(data)+.5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Bleu score distribution')
plt.xlabel('Bleu score')
plt.ylabel('count')
plt.show()


#calling test data
m = final_data(hp.vocab, hp.test1, hp.test1, hp.maxlen1, hp.maxlen2)
x_test, _, _ = m.compute()

#predicting
s_id_test, sentence_test, output_sentence_test = decode(
                                          model,
                                          x_test,
                                          dict_token2idx = m.token2idx,
                                          dict_idx2token = m.idx2token,
                                          sp = sp
                                          )

#writing predicted output in file
with open(hp.test_output_file, "w") as fout:
    for sent in output_sentence_test:
        fout.write(sent + "\n")
