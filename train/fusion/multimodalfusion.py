# imports
import math
import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
from transformers import BertTokenizer
# from models.bert import BERTModel
# from models.fer import FERModel

''' Load models in ? They are from different files '''
# get info from file
pathToFER = '../facialexpressionrecognition.ipynb' # assumed for now
pathToBERT = '../bertforsentimentanalysis.ipynb'

# load in BERT     # returns categorization -- [1, 0]
# load in FER model
    # def load_fer_model(): return FERModel()
    # returns probability, thresholded to 0 or 1

fer_classes = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear', 'neutral']
bert_classes = ['positive', 'negative'] # 1 or 0

'''  
providing 'soft' probabilities for BERT-sentiment-FER outputs
BERT outputs {0, 1} (2 classes) | FER outputs {happiness ... sadness} (7 classes)
'''
tone_to_face = {
    1: {
        'happiness': 1.0,
        'surprise': 0.8,
        'neutral': 0.6,
        'sadness': 0.3,
        'anger': 0.2,
        'disgust': 0.1,
        'fear': 0.2
    },
    0: {
        'happiness': 0.2,
        'surprise': 0.4,
        'neutral': 0.5,
        'sadness': 1.0,
        'anger': 0.8,
        'disgust': 0.7,
        'fear': 0.6
    }
}
''' Desired output of system: -1 is negative; 1 is positive, 0 is neutral'''

''' Get FER and BERT output classification '''
def preprocessing(bert, fer):
    BERT_out = bert.get_probabilities() # get_probabilities is in external file
    FER_out = fer.get_probabilities()
    return BERT_out, FER_out

''' DECISION LEVEL MULTIMODAL DATA FUSION SCRIPT '''
# Late fusion model script
class late_fusion_model(nn.Module):
    def __init__(self): 
        super(late_fusion_model, self).__init__()
        
        ''' FER was mapped to 3 classes, so its (2 + 3) = (5) input dimensions'''
        self.fc = nn.Linear(9, 3) # nn.Linear(input dimensions, output dimensions desired)
        self.softmax = nn.Softmax(dim=1) # probability distribution

    def forward(self, bert_pred, fer_pred):
        ''' 
        BERT: tensor [1, 2]
        FER: tensor [1, 7]
        '''

        sentiment_class = int(torch.argmax(bert_pred, dim=1).item()) # max class after softmax layer
        class_from_map = tone_to_face[sentiment_class] # enter dominant tone thing

        # applying weights of BERT mapping to FER
        fer = []
        for i, cls in enumerate(fer_classes):
            fer.append(fer_pred[0, i].item() * class_from_map[cls])
        fer = torch.tensor([fer], dtype=fer_pred.dtype, device=fer_pred.device) 

        input = torch.cat([bert_pred, fer], dim=1) # [1,5] final classification layer

        ''' Pre-Processing + One-Hot Encoding '''
        ''' Tensor will have shape 3, with probabilities per classification '''
        sentiment = self.fc(input)
        sentiment = self.softmax(sentiment)

        return sentiment

''' main(?) for inference 
BERT_out, FER_out = preprocessing(BERT_get_out, FER_get_out) # At this point, we have the output classification. 

# Convert to tensors for input into model (Expects 5 (2 from BERT, 3 from FER))
bert_tensor = torch.tensor(BERT_out)
FER_tensor = torch.tensor(FER_out)

'''