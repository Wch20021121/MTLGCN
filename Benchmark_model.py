import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class MultitaskGCN(nn.Module):
    def __init__(self, tags_size, num_sentiment_classes, all_sentiment_classes, p0, bert_model, hidden_dim):
        super(MultitaskGCN, self).__init__()
        # 添加 BERT 模型
        self.bert = bert_model
        # 定义一个bi-lstm层,输入用的是上层bert的输出
        self.bi_lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim // 2,
                               num_layers=2, bidirectional=True, batch_first=True)
        # 定义一个分类器
        self.hidden2tag = nn.Linear(hidden_dim, tags_size)
        # 定义一个CRF层
        self.crf = CRF(tags_size, batch_first=True)
        # bert模型的输出特征
        self.conv1 = GCNConv(self.bert.config.hidden_size, 16)
        self.conv2 = GCNConv(16, 16)
        self.sentiment_classifier = Linear(16, num_sentiment_classes)
        self.all_classifier = Linear(16, all_sentiment_classes)
        self.dropout = nn.Dropout(p=p0)
        self.p0 = p0

    def forward(self, data):
        # 首先通过 BERT 模型处理输入数据
        outputs = self.bert(data.input_ids, attention_mask=data.attention_mask)
        x = outputs.last_hidden_state
        # 通过bi-lstm层
        lstm_output, _ = self.bi_lstm(x)
        lstm_output = self.dropout(lstm_output)
        lstm_feats = self.hidden2tag(lstm_output)
        # 计算CRF的loss
        global entity_loss
        if data.training:
            # 把data.entity的维度从tensor[节点数]变成Tensor[1,节点数]
            entity = data.entity.unsqueeze(0)
            # 输出是否是实体的loss
            entity_loss = -self.crf(lstm_feats, entity, mask=data.attention_mask.byte(), reduction='mean')
        # 输出实体的概率
        entity_output = self.crf.decode(lstm_feats, mask=data.attention_mask.byte())[0]
        edge_index = data.edge_index
        # 把x转换成二维[节点数，特征数],且长度转换成节点数
        # 去掉x的第一维使其从[batch,节点数，特征数]变成[节点数，特征数],batch=1
        x = x.squeeze(0)
        # 把x的维度转换成[节点数，特征数]
        if data.total_length >= 512:
            x = x[1:, :]
        else:
            x = x[1:data.total_length + 1, :]
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=self.p0, training=data.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # 输出所有节点的情感
        sentiment_output = self.sentiment_classifier(x)

        # 输出整体句子的情感
        all_output = global_mean_pool(x, data.batch)
        all_output = self.all_classifier(all_output)

        # 删去entity_output中bert编码的特殊符号[CLS]和[SEP]
        entity_output = entity_output[1:data.total_length+1]

        # 只保留实体节点的情感输出,实体节点的entity标签为1和2
        y = torch.tensor(entity_output, dtype=torch.long)
        entity_mask = y != 0  # 实体类别不为0的节点
        sentiment_output = sentiment_output[entity_mask]

        if data.training:
            return (entity_loss, entity_output,
                    F.log_softmax(sentiment_output, dim=1),
                    F.log_softmax(all_output, dim=1),)
        else:
            return entity_output, F.softmax(sentiment_output, dim=1), F.softmax(all_output, dim=1)
