from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import GCNConv
import torch
from torch_geometric.nn.glob import global_mean_pool
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


# 多任务GAT模型
class MultitaskGAT(nn.Module):
    def __init__(self, tags_size, num_sentiment_classes, all_sentiment_classes, p0, bert_model, hidden_dim):
        super(MultitaskGAT, self).__init__()
        # 添加 BERT 模型
        self.bert = bert_model
        # 定义一个bi-lstm层,输入用的是上层bert的输出
        self.bi_lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim // 2,
                               num_layers=2, bidirectional=True, batch_first=True)
        # 定义一个CRF层
        self.crf = CRF(tags_size, batch_first=True)
        # bert模型的输出特征
        self.conv1 = GATConv(hidden_dim, 256)
        self.conv2 = GATConv(256, 256)
        self.sentiment_classifier = Linear(256, num_sentiment_classes)
        self.all_classifier = Linear(256, all_sentiment_classes)
        # 定义一个分类器
        self.hidden2tag = nn.Linear(256, tags_size)
        self.dropout = nn.Dropout(p=p0)
        self.p0 = p0

    def forward(self, data):
        # 首先通过 BERT 模型处理输入数据
        global entity_loss
        outputs = self.bert(data.input_ids, attention_mask=data.attention_mask)
        bert_output = outputs.last_hidden_state
        # 通过bi-lstm层
        lstm_output, _ = self.bi_lstm(bert_output)
        lstm_output = self.dropout(lstm_output)
        lstm_output = lstm_output.squeeze(0)
        # 处理lstm_output使其长度为节点数,去掉[CLS]和[SEP]，再修改mask的长度使其与lstm_output对应
        if data.total_length >= 512:
            # 去掉[CLS]和[SEP]
            lstm_output = lstm_output[1:, :]
            attention_mask = data.attention_mask[:, 1:]
        else:
            lstm_output = lstm_output[1:data.total_length + 1, :]
            attention_mask = data.attention_mask[:, 1:data.total_length + 1]
        # 通过GCN
        conv_output = self.conv1(lstm_output, data.edge_index)
        conv_output = torch.relu(conv_output)
        conv_output = F.dropout(conv_output, p=self.p0, training=data.training)
        conv_output = self.conv2(conv_output, data.edge_index)
        conv_output = torch.relu(conv_output)
        # 输出所有节点的情感
        sentiment_output = self.sentiment_classifier(conv_output)
        # 输出整体句子的情感
        all_output = global_mean_pool(conv_output, data.batch)
        all_output = self.all_classifier(all_output)
        # 计算CRF的loss
        # 给conv_output添加一个维度
        conv_output = conv_output.unsqueeze(0)
        lstm_feats = self.hidden2tag(conv_output)
        if data.training:
            # 增加data.entity的维度
            entity = data.entity.unsqueeze(0)
            entity_loss = -self.crf(lstm_feats, entity, mask=attention_mask.byte(), reduction='mean')
        # 输出实体的概率
        entity_output = self.crf.decode(lstm_feats, mask=attention_mask.byte())[0]

        # 只保留实体节点的情感输出,实体节点的entity标签不为0
        y = torch.tensor(entity_output, dtype=torch.long)
        entity_mask = y != 0  # 实体类别不为0的节点
        sentiment_output = sentiment_output[entity_mask]

        if data.training:
            return (entity_loss, entity_output,
                    F.log_softmax(sentiment_output, dim=1),
                    F.log_softmax(all_output, dim=1),)
        else:
            return entity_output, F.softmax(sentiment_output, dim=1), F.softmax(all_output, dim=1)


# 多任务GCN模型
class MultitaskGCN(nn.Module):
    def __init__(self, tags_size, num_sentiment_classes, all_sentiment_classes, p0, bert_model, hidden_dim):
        super(MultitaskGCN, self).__init__()
        # 添加 BERT 模型
        self.bert = bert_model
        # 定义一个bi-lstm层,输入用的是上层bert的输出
        self.bi_lstm = nn.LSTM(bert_model.config.hidden_size, hidden_dim // 2,
                               num_layers=2, bidirectional=True, batch_first=True)
        # 定义一个CRF层
        self.crf = CRF(tags_size, batch_first=True)
        # bert模型的输出特征
        self.conv1 = GCNConv(hidden_dim, 256)
        self.conv2 = GCNConv(256, 256)
        self.sentiment_classifier = Linear(256, num_sentiment_classes)
        self.all_classifier = Linear(256, all_sentiment_classes)
        # 定义一个分类器
        self.hidden2tag = nn.Linear(256, tags_size)
        self.dropout = nn.Dropout(p=p0)
        self.p0 = p0

    def forward(self, data):
        # 首先通过 BERT 模型处理输入数据
        global entity_loss
        outputs = self.bert(data.input_ids, attention_mask=data.attention_mask)
        bert_output = outputs.last_hidden_state
        # 通过bi-lstm层
        lstm_output, _ = self.bi_lstm(bert_output)
        lstm_output = self.dropout(lstm_output)
        lstm_output = lstm_output.squeeze(0)
        # 处理lstm_output使其长度为节点数,去掉[CLS]和[SEP]，再修改mask的长度使其与lstm_output对应
        if data.total_length >= 512:
            # 去掉[CLS]和[SEP]
            lstm_output = lstm_output[1:, :]
            attention_mask = data.attention_mask[:, 1:]
        else:
            lstm_output = lstm_output[1:data.total_length + 1, :]
            attention_mask = data.attention_mask[:, 1:data.total_length + 1]
        # 通过GCN
        conv_output = self.conv1(lstm_output, data.edge_index)
        conv_output = torch.relu(conv_output)
        conv_output = F.dropout(conv_output, p=self.p0, training=data.training)
        conv_output = self.conv2(conv_output, data.edge_index)
        conv_output = torch.relu(conv_output)
        # 输出所有节点的情感
        sentiment_output = self.sentiment_classifier(conv_output)
        # 输出整体句子的情感
        all_output = global_mean_pool(conv_output, data.batch)
        all_output = self.all_classifier(all_output)
        # 计算CRF的loss
        # 给conv_output添加一个维度
        conv_output = conv_output.unsqueeze(0)
        lstm_feats = self.hidden2tag(conv_output)
        if data.training:
            # 增加data.entity的维度
            entity = data.entity.unsqueeze(0)
            entity_loss = -self.crf(lstm_feats, entity, mask=attention_mask.byte(), reduction='mean')
        # 输出实体的概率
        entity_output = self.crf.decode(lstm_feats, mask=attention_mask.byte())[0]

        # 只保留实体节点的情感输出,实体节点的entity标签不为0
        y = torch.tensor(entity_output, dtype=torch.long)
        entity_mask = y != 0  # 实体类别不为0的节点
        sentiment_output = sentiment_output[entity_mask]

        if data.training:
            return (entity_loss, entity_output,
                    F.log_softmax(sentiment_output, dim=1),
                    F.log_softmax(all_output, dim=1),)
        else:
            return entity_output, F.softmax(sentiment_output, dim=1), F.softmax(all_output, dim=1)
