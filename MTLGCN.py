import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


# 多任务方面级情感分析的GCN模型
class MultitaskGCN(nn.Module):
    def __init__(self, num_node_features, num_entity_classes, num_sentiment_classes, all_sentiment_classes):
        super(MultitaskGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.entity_classifier = Linear(16, num_entity_classes)
        self.sentiment_classifier = Linear(16, num_sentiment_classes)
        self.all_classifier = Linear(16, all_sentiment_classes)
        self.training = True

    def forward(self, data):
        x, edge_index = data.node_feature, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        # 输出是否是实体的概率
        entity_output = self.entity_classifier(x)
        # 输出所有节点的情感
        sentiment_output = self.sentiment_classifier(x)

        # 输出整体句子的情感
        all_output = global_mean_pool(x, data.batch)
        all_output = self.all_classifier(all_output)

        # 只保留实体节点的情感输出
        entity_mask = torch.argmax(entity_output, dim=1) == 1  # 假设实体类别为1
        sentiment_output = sentiment_output[entity_mask]

        return (F.log_softmax(entity_output, dim=1),
                F.log_softmax(sentiment_output, dim=1),
                F.log_softmax(all_output, dim=1),
                entity_mask)


