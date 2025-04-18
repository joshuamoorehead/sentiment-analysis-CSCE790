# Facial Expression Recognition at the Edge

1. Facial Expression Recognition (FER) - PAtt-Lite Model
2. Sentiment Analysis - BERT
3. Data Fusion of FER and Sentiment Analysis Networks

---

## 1. Facial Expression Recognition (FER)

We use a lightweight PAtt-Lite convolutional neural network trained on the FER2013 dataset for expression recognition. The model is designed for edge inference on devices like the Raspberry Pi 5.

- Training script: `train/train_patt_lite.py`
- Model architecture: `models/patt_lite.py`
- Inference script: `infer_patt_lite.py`



## 2. Sentiment Analysis (BERT)

The sentiment analysis is done using a fine-tuned BERT model with a custom classification head.

- Training notebook: `train/bertforsentimentanalysis.ipynb`
- Model architecture: `models/bert.py`
- Inference script `run_sentiment_analysis.py`

We fine-tuned BERT on sentiment analysis datasets including Archeage, Ntua, and HCR.

---

## 3. Multimodal Fusion

The FER and sentiment models are combined in `multimodalfusion.py` to create a joint expression/sentiment decision model. Future iterations aim to improve fusion latency and decision thresholds for real-time applications.

---

## Install

Developed for Raspberry Pi 5:

```bash
sudo apt install portaudio19-dev
sudo apt install flac
```

---

## Sources

* https://raspberrypi.stackexchange.com/questions/84666/problem-on-installing-pyaudio-on-raspberry-pi
* https://github.com/kpolley/Python_AVrecorder/blob/master/picam.py
* https://github.com/JLREX/PAtt-Lite
