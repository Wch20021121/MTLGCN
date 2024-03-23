from ltp import LTP
from transformers import BertTokenizer
import json
import torch
import logging
from tqdm import tqdm


# 读取并处理数据||read and processing data
def read_process_data(file_name):
    path = './process_data/'
    # 读取处理好的json数据
    with open(path + file_name, 'r', encoding='utf-8') as f:
        use_data = json.load(f)
        f.close()
    return use_data


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
    bert_input = []
    edge_index = []
    batch = []
    y_entity = []
    y_sentiment = []
    all_sentiment = []
    total_length = []
    for key in tqdm(data, desc="正在处理ltp数据", ncols=100, total=len(data), dynamic_ncols=True):
        # 获取节点特征
        bert_token = tokenizer.encode_plus(key, max_length=512, padding='max_length',
                                           return_attention_mask=True, add_special_tokens=True,
                                           return_token_type_ids=True, return_tensors='pt', truncation=True)
        bert_input.append(bert_token)
        total_length.append(len(key))
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
                entity[j] = 2
                sentiment[j] = int(data[key][i + 2]) + 2
            entity[index[0]] = 1
        # 去除sentiment中的5
        for i in range(len(sentiment) - 1, -1, -1):
            if sentiment[i] == 5:
                sentiment.pop(i)
        y_entity.append(entity)
        y_sentiment.append(sentiment)
    return bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length


def get_matrix_data_adjacency(data):
    # 定义bert分词器,这里使用的是中文的bert模型
    tokenizer = BertTokenizer.from_pretrained(r'D:\Model\transformers\text-classier\bert_model\bert-base-chinese')
    # 定义所有需要的数据
    emotion_dict = {}
    bert_input = []
    edge_index = []
    batch = []
    y_entity = []
    y_sentiment = []
    all_sentiment = []
    total_length = []
    for key in tqdm(data, desc="正在处理adjacency数据", ncols=100, total=len(data), dynamic_ncols=True):
        # 获取节点特征
        bert_token = tokenizer.encode_plus(key, max_length=512, padding='max_length',
                                           return_attention_mask=True, add_special_tokens=True,
                                           return_token_type_ids=True, return_tensors='pt', truncation=True)

        bert_input.append(bert_token)
        total_length.append(len(key))
        # 获取batch（批次）
        batch.append([0] * len(key))
        # 整体情感标记
        # -1的情感标记变成0
        if int(data[key][0]) == -1:
            all_sentiment.append([data[key][0] + 1])
        else:
            all_sentiment.append([data[key][0]])
        # 获取实体标记和情感标记，-2到2把它变成0到4
        entity = [0] * len(key)
        sentiment = [5] * len(key)
        # 创建边索引值,这里使用的是邻接矩阵，和情感字典
        edge = [[], []]
        for i in range(1, len(data[key]), 3):
            index = find_index(key, data[key][i])
            # 如果情感字典中没有这个情感，就添加进去
            if data[key][i + 1] not in emotion_dict:
                emotion_dict[data[key][i + 1]] = [data[key][i]]
            else:
                if data[key][i] not in emotion_dict[data[key][i + 1]]:
                    emotion_dict[data[key][i + 1]].append(data[key][i])
            for j in index:
                entity[j] = 2
                sentiment[j] = int(data[key][i + 2]) + 2
                edge[0].append(j)
                edge[1].append(j)
            entity[index[0]] = 1
        # 去除sentiment中的5
        for i in range(len(sentiment) - 1, -1, -1):
            if sentiment[i] == 5:
                sentiment.pop(i)
        edge_index.append(edge)
        y_entity.append(entity)
        y_sentiment.append(sentiment)
    return bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, emotion_dict, total_length


def save_dict(emotion_dict, path):
    with open(path + '/' + 'emotion_dict.json', 'w', encoding='utf-8') as f:
        json.dump(emotion_dict, f, ensure_ascii=False)
        f.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 获取训练数据
    train_org_data = read_process_data('train_data.json')
    # 获取测试数据
    test_org_data = read_process_data('test_data.json')

    logging.info("正在处理ltp的数据:")
    bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_matrix_data_ltp(
        train_org_data)
    train_data_ltp = (bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length)
    # 获取测试数据
    bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_matrix_data_ltp(
        test_org_data)
    test_data_ltp = (bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length)
    # 获取训练数据
    logging.info("正在处理adjacency的数据:")
    (bert_input, edge_index, batch, y_entity,
     y_sentiment, all_sentiment, emotion_dict, total_length) = get_matrix_data_adjacency(train_org_data)
    train_data_adjacency = (bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, emotion_dict,
                            total_length)
    # 获取测试数据
    (bert_input, edge_index, batch, y_entity,
     y_sentiment, all_sentiment, emotion_dict, total_length) = get_matrix_data_adjacency(test_org_data)
    test_data_adjacency = (bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, emotion_dict,
                           total_length)
