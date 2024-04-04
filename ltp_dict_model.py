from torch_geometric.nn.conv import GATConv
import torch
from torch_geometric.nn.glob import global_mean_pool
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


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
        self.sentiment_classifier = Linear(256*2, num_sentiment_classes)
        self.all_classifier = Linear(256*2, all_sentiment_classes)
        # 定义一个分类器
        self.hidden2tag = nn.Linear(256*2, tags_size)
        self.dropout = nn.Dropout(p=p0)
        self.p0 = p0

    def forward(self, data_ltp, data_dict):
        # 首先通过 BERT 模型处理输入数据
        global entity_loss
        outputs = self.bert(data_ltp.input_ids, attention_mask=data_ltp.attention_mask)
        bert_output = outputs.last_hidden_state
        # 通过bi-lstm层
        lstm_output, _ = self.bi_lstm(bert_output)
        lstm_output = self.dropout(lstm_output)
        lstm_output = lstm_output.squeeze(0)
        # 处理lstm_output使其长度为节点数,去掉[CLS]和[SEP]，再修改mask的长度使其与lstm_output对应
        if data_ltp.total_length >= 512:
            # 去掉[CLS]和[SEP]
            lstm_output = lstm_output[1:, :]
            attention_mask = data_ltp.attention_mask[:, 1:]
        else:
            lstm_output = lstm_output[1:data_ltp.total_length + 1, :]
            attention_mask = data_ltp.attention_mask[:, 1:data_ltp.total_length + 1]
        # 通过ltp_GCN
        conv_output_ltp = self.conv1(lstm_output, data_ltp.edge_index)
        conv_output_ltp = torch.relu(conv_output_ltp)
        conv_output_ltp = F.dropout(conv_output_ltp, p=self.p0, training=data_ltp.training)
        conv_output_ltp = self.conv2(conv_output_ltp, data_ltp.edge_index)
        conv_output_ltp = torch.relu(conv_output_ltp)
        # 通过dict_GCN
        conv_output_dict = self.conv1(lstm_output, data_dict.edge_index)
        conv_output_dict = torch.relu(conv_output_dict)
        conv_output_dict = F.dropout(conv_output_dict, p=self.p0, training=data_dict.training)
        conv_output_dict = self.conv2(conv_output_dict, data_dict.edge_index)
        conv_output_dict = torch.relu(conv_output_dict)
        # 把con_output_ltp和conv_output_dict合并,维度从[num_nodes, 256]变为[num_nodes, 512]
        conv_output = torch.cat((conv_output_ltp, conv_output_dict), dim=1)
        # 输出所有节点的情感
        sentiment_output = self.sentiment_classifier(conv_output)
        # 输出整体句子的情感
        all_output = global_mean_pool(conv_output, data_ltp.batch)
        all_output = self.all_classifier(all_output)
        # 计算CRF的loss
        # 给conv_output添加一个维度
        conv_output = conv_output.unsqueeze(0)
        lstm_feats = self.hidden2tag(conv_output)
        if data_ltp.training:
            # 增加data.entity的维度
            entity = data_ltp.entity.unsqueeze(0)
            entity_loss = -self.crf(lstm_feats, entity, mask=attention_mask.byte(), reduction='mean')
        # 输出实体的概率
        entity_output = self.crf.decode(lstm_feats, mask=attention_mask.byte())[0]

        # 只保留实体节点的情感输出,实体节点的entity标签不为0
        y = torch.tensor(entity_output, dtype=torch.long)
        entity_mask = y != 0  # 实体类别不为0的节点
        sentiment_output = sentiment_output[entity_mask]

        if data_ltp.training:
            return (entity_loss, entity_output,
                    F.log_softmax(sentiment_output, dim=1),
                    F.log_softmax(all_output, dim=1),)
        else:
            return entity_output, F.softmax(sentiment_output, dim=1), F.softmax(all_output, dim=1)
