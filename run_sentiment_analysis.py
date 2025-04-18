import torch
from transformers import BertTokenizer
from models.bert import BERTForSentimentAnalysis
from transformers.models.bert.configuration_bert import BertConfig

class SentimentAnalysis:
    def __init__(self, debug=False):
        self.config = BertConfig()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.DEBUG = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path):
        model = BERTForSentimentAnalysis(self.config).from_pretrained('bert-base-cased')
        #if self.device == 'cuda':
        #    model.load_state_dict(torch.load(model_path, weights_only=True))
        #elif self.device == 'cpu':
        model.load_state_dict(torch.load(model_path,
                                         weights_only=True,
                                         map_location=torch.device(self.device)))
        model.to(self.device)
        model.eval()

        return model

    def prepare_input_data(self, data):
        data = " ".join(str(data).split())
        data = self.tokenizer.encode_plus(data, None,
                                                    add_special_tokens=True,
                                                    max_length=self.config.max_length,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True)
        ids = torch.tensor(data['input_ids'], dtype=torch.long)
        ids = ids.to(self.device, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(data['attention_mask'], dtype=torch.long)
        mask = mask.to(self.device, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(data['token_type_ids'], dtype=torch.long)
        token_type_ids = token_type_ids.to(self.device, dtype=torch.long).unsqueeze(0)

        return ids, mask, token_type_ids

    def inference(self, model, data):
        if self.DEBUG:
            print('inferencing on data:', data)

        with torch.no_grad():
            ids, mask, token_type_ids = self.prepare_input_data(data)
            output = model(ids, mask, token_type_ids)
            prediction = torch.argmax(output, 1).cpu()
            print('prediction:', prediction)

        return prediction


if __name__ == "__main__":
    sentimentanalysis = SentimentAnalysis(True)
    model = sentimentanalysis.load_model('./weights/model.pt')
    sentimentanalysis.inference(model, "no, im so disappointed")
