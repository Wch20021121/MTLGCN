import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


# 多任务方面级情感分析的GCN模型
class MultitaskGCN(nn.Module):
    def __init__(self, num_node_features, num_entity_classes, num_sentiment_classes):
        super(MultitaskGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.entity_classifier = Linear(16, num_entity_classes)
        self.sentiment_classifier = Linear(16, num_sentiment_classes)
        self.training = True

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        entity_output = self.entity_classifier(x)
        sentiment_output = self.sentiment_classifier(x)

        # 输出整体句子的情感
        all_output = global_mean_pool(sentiment_output, data.batch)

        # 只保留实体节点的情感输出
        entity_mask = torch.argmax(entity_output, dim=1) == 1  # 假设实体类别为1
        sentiment_output = sentiment_output[entity_mask]

        return F.log_softmax(entity_output, dim=1), F.log_softmax(sentiment_output, dim=1), F.log_softmax(all_output, dim=1)
