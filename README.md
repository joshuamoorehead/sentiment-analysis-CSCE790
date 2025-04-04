# Facial Expression Recognition at the Edge
1. Facial Expression Recognition (FER)
2. Sentiment Analysis - BERT
3. Data fusion of FER and sentiment analysis networks

## Sentiment Analysis
The sentiment analysis is done with a BERT model. The [training script](train/bertforsentimentanalysis.ipynb) uses a pretrained BERT model and a custom classification head. The model is fine-tuned on sentiment analysis datasets.

## Install
```
$ sudo apt install portaudio19-dev
$ sudo apt install flac
```
[src](https://raspberrypi.stackexchange.com/questions/84666/problem-on-installing-pyaudio-on-raspberry-pi)

## Sources
* https://github.com/kpolley/Python_AVrecorder/blob/master/picam.py

