
class config_set():
    """
    contains all the features required in the model.
     maxlen1 = input sequence max sequence length
     maxlen2 = output sequence max sequence length
     train1 = training data path for german text
     train2 = training data path for english text
     eval1 = evaluation data path for german text
     eval2 = evaluation data path for english text
     test1 = testing data path for german text
     test2 = testing data path for english text
     vocab = path of vocab file where all the tokens are listed from training data
     embed_model = path of tokenizing model used (sentencepiece)
     vocab_size = vocab size
     batch_size = batch size
     sample_num_display = translated sentence (actual vs pred) to show while scoring
     predicted_output_file = path for storing predicted output document for eval set
     actual_output_file = path for storing actual output document for eval set
     test_output_file = path for storing predicted output document for test set
    """
    def __init__(self):
        self.maxlen1 = 20
        self.maxlen2 = 20
        self.train1 = "iwslt2016/segmented/train.de.bpe"
        self.train2 = "iwslt2016/segmented/train.en.bpe"
        self.eval1 = "iwslt2016/segmented/eval.de.bpe"
        self.eval2 = "iwslt2016/segmented/eval.en.bpe"
        self.test1 = "iwslt2016/segmented/test.de.bpe"
        self.test2 = "iwslt2016/segmented/test.en.bpe"
        self.vocab = "iwslt2016/segmented/bpe.vocab"
        self.embed_model = "iwslt2016/segmented/bpe.model"
        self.vocab_size = 32000
        self.batch_size = 16
        self.sample_num_display = 10
        self.predicted_output_file = "iwslt2016/segmented/model_predictions"
        self.actual_output_file = "iwslt2016/segmented/model_actuals"
        self.test_output_file = "iwslt2016/segmented/model_test_output"
