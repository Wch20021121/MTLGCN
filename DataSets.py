import torch
from torch_geometric.data import Data


# 创建需要的x，edge_index，和batch
class MyDataSet:
    def __init__(self, node_feature, edge_index, batch, y_entity, y_sentiment):
        self.x = node_feature
        self.edge_index = edge_index
        self.batch = batch
        self.y_entity = y_entity
        self.y_sentiment = y_sentiment

    def data_embedding(self):
        datalist = []
        for i in range(len(self.x)):
            x_i = torch.tensor(self.x[i], dtype=torch.float)
            edge_index_i = torch.tensor(self.edge_index[i], dtype=torch.long)
            batch_i = torch.tensor(self.batch[i], dtype=torch.long)
            y_entity_i = torch.tensor(self.y_entity[i], dtype=torch.long)
            y_sentiment_i = torch.tensor(self.y_sentiment[i], dtype=torch.long)
            data = Data(x=x_i, edge_index=edge_index_i, batch=batch_i, y_entity=y_entity_i, y_sentiment=y_sentiment_i)
            datalist.append(data)
        return datalist
