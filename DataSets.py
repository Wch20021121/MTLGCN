import torch
from torch_geometric.data import Data
from tqdm import tqdm


# 创建需要的x，edge_index，和batch
class MyDataSet:
    def __init__(self, bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length, training):
        self.bert_input = bert_input
        self.edge_index = edge_index
        self.batch = batch
        self.y_entity = y_entity
        self.y_sentiment = y_sentiment
        self.all_sentiment = all_sentiment
        self.total_length = total_length
        self.training = training

    def data_embedding(self):
        datalist = []
        for i in tqdm(range(len(self.bert_input)), desc="Data embedding", ncols=100, total=len(self.bert_input),
                      dynamic_ncols=True):
            input_ids_i = self.bert_input[i]['input_ids']
            attention_mask_i = self.bert_input[i]['attention_mask']
            edge_index_i = torch.tensor(self.edge_index[i], dtype=torch.long)
            batch_i = torch.tensor(self.batch[i], dtype=torch.long)
            if self.training:
                y_entity_i = [0]+self.y_entity[i]+[0]*(512-len(self.y_entity[i])-1)
                y_entity_i = torch.tensor(y_entity_i, dtype=torch.long)
            else:
                y_entity_i = self.y_entity[i]
            y_sentiment_i = torch.tensor(self.y_sentiment[i], dtype=torch.long)
            all_sentiment_i = torch.tensor(self.all_sentiment[i], dtype=torch.long)
            total_length_i = self.total_length[i]
            data = Data(input_ids=input_ids_i, attention_mask=attention_mask_i, edge_index=edge_index_i, batch=batch_i,
                        entity=y_entity_i, aspect_sentiment=y_sentiment_i, all_sentiment=all_sentiment_i,
                        total_length=total_length_i, training=self.training)
            datalist.append(data)
        return datalist
