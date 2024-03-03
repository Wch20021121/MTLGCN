from ltp import LTP
from transformers import BertTokenizer
import json
import torch


# 读取并处理数据||read and processing data
def read_process_data():
    # 读取处理好的json数据
    with open('./process_data/same_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
        x = []
        for key in data:
            x.append([key] + data[key])
        # 去除语句中的空格和方面中的空格
        for i in range(len(x)):
            x[i][0] = x[i][0].replace(' ', '')
            for j in range(2, len(x[i]), 3):
                x[i][j] = x[i][j].replace(' ', '')
        data = {}
        for i in range(len(x)):
            data[x[i][0]] = x[i][1:]
    return data


def find_index(text, aspect):
    if aspect in text:
        a = [text.index(aspect)]
        for i in range(len(aspect) - 1):
            a.append(a[i] + 1)
        return a
    else:
        a = []
        for i in range(len(aspect)):
            if aspect[i] in text:
                a.append(text.index(aspect[i]))
        return a


# 只根据LTP获取边链接关系
def get_matrix_data_ltp(data):
    # data 是字典格式{text:[总分，实体，方面，情感----]}（情感是分为5类：-2到2）
    # 定义bert分词器
    tokenizer = BertTokenizer.from_pretrained(r'D:\Model\transformers\text-classier\bert_model\bert-base-chinese')
    # 定义ltp分词器
    ltp = LTP(pretrained_model_name_or_path=r'D:\大创项目\MTLGCN\LTP_model')
    # 把ltp移到GPU上
    if torch.cuda.is_available():
        ltp.to("cuda")
    # 定义所有需要的数据
    node_feature = []
    edge_index = []
    batch = []
    y_entity = []
    y_sentiment = []
    all_sentiment = []
    for key in data:
        # 获取节点特征
        node_feature_i = []
        bert_token = tokenizer.encode_plus(key, add_special_tokens=False)['input_ids']
        for i in bert_token:
            node_feature_i.append([i])
        if len(node_feature_i) != len(key):
            # 完善节点特征，没有的特征值用0完善
            node_feature_i += [[0]] * (len(key) - len(node_feature[-1]))
        node_feature.append(node_feature_i)
        # 获取batch（批次）
        batch.append([0] * len(key))
        # 读取数据
        output = ltp.pipeline(key, tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
        # 获取分词结果
        text = output.cws
        # 获取依存句法分析树(图)
        sdpg = output.sdpg
        # 创建边索引值
        edge = [[], []]
        for i in range(len(sdpg)):
            for j in range(len(text[sdpg[i][0] - 1])):
                if sdpg[i][1] == 0:
                    break
                for z in range(len(text[sdpg[i][1] - 1])):
                    # 根据LTP的依存句法分析树，获取边索引值
                    edge[0].append(key.index(text[sdpg[i][0] - 1]) + j)
                    edge[1].append(key.index(text[sdpg[i][1] - 1]) + z)
        edge_index.append(edge)
        # 整体情感标记
        # -1的情感标记变成0
        if int(data[key][0]) == -1:
            all_sentiment.append([data[key][0] + 1])
        else:
            all_sentiment.append([data[key][0]])
        # 获取实体标记和情感标记，-2到2把它变成0到4
        entity = [0] * len(key)
        sentiment = [5] * len(key)
        for i in range(1, len(data[key]), 3):
            index = find_index(key, data[key][i])
            for j in index:
                entity[j] = 1
                sentiment[j] = int(data[key][i + 2]) + 2
        y_entity.append(entity)
        y_sentiment.append(sentiment)
    return node_feature, edge_index, batch, y_entity, y_sentiment, all_sentiment


if __name__ == '__main__':
    data = read_process_data()
    print("================================正在处理数据================================")
    node_feature, edge_index, batch, y_entity, y_sentiment, all_sentiment = get_matrix_data_ltp(data)
    print("================================数据处理完成================================")

