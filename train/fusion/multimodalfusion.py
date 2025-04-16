# imports
import math
import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
from transformers import BertTokenizer
# from bertforsentimentanalysis


''' Load models in ? They are from different files '''
# get info from file
pathToFER = '../facialexpressionrecognition.ipynb' # assumed for now
pathToBERT = '../bertforsentimentanalysis.ipynb'

# load in BERT     # returns categorization -- 'positive' (1), 'negative' (0)
# load in FER model
    # def load_fer_model(): return FERModel()
    # returns probability, thresholded to 0 or 1

'''  
mapping BERT sentiment to FER outputs
BERT outputs {negative, positive} | FER outputs {happiness ... sadness} (7 classes)
-1 is negative; 1 is positive, 0 is neutral
'''
tone_to_face = {
    "happiness" :   1,
    "surprise" :    0,
    "neutral" :     0,
    "sadness" :    -1,
    "anger" :      -1,
    "disgust" :    -1,
    "fear" :       -1
}

# TODO
def preprocessing(bert, fer):
    ''' Get BERT output classification '''
    BERT_out = bert.get # is it string or integer  ? 
    '''' Get FER output classification '''
    FER_out = fer.get()

    return BERT_out, FER_out


''' DECISION LEVEL MULTIMODAL DATA FUSION SCRIPT '''
# Late fusion model script
class late_fusion_model(nn.Module):
    def __init__(self): 
        super(late_fusion_model, self).__init__()
        
        ''' FER was mapped to 3 classes, so its (2 + 3) = (5) input dimensions'''
        self.fc = nn.Linear(5, 3) # nn.Linear(input dimensions, output dimensions desired)
        self.softmax = nn.Softmax(dim=1) # probability distribution

    def forward(self, bert_pred, fer_pred):
        
        ''' Pre-Processing + One-Hot Encoding '''


        ''' Tensor will have shape 3, with probabilities per classification '''
        sentiment = self.fc(input)
        sentiment = self.softmax(sentiment)

        return sentiment

''' main(?) for inference ''' 
BERT_out, FER_out = preprocessing(BERT_get_out, FER_get_out) # At this point, we have the output classification. 

# Convert to tensors for input into model (Expects 5 (2 from BERT, 3 from FER))
bert_tensor = torch.tensor(BERT_out)
FER_tensor = torch.tensor(FER_out)

