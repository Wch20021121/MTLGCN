from torch.nn import CrossEntropyLoss
from base_model import MultitaskGAT, MultitaskGCN
from DataSets_bblg import MyDataSet
from torch.optim.lr_scheduler import OneCycleLR
from DataProcess import *
from transformers import BertModel
import warnings
from torch.optim import AdamW
from config import *


# noinspection PyShadowingNames
def train_model(train_dataloader, tags_size, num_sentiment_classes, all_sentiment_classes,
                num_epochs, learning_rate, device, weight, p0, accumulation_steps, bert_model, hidden_dim,
                sentiment_loss_weight, all_sentiment_loss_weight, model_type):
    # 初始化模型和优化器
    if model_type == 'GCN':
        model = MultitaskGCN(tags_size, num_sentiment_classes, all_sentiment_classes, p0, bert_model, hidden_dim)
    else:
        model = MultitaskGAT(tags_size, num_sentiment_classes, all_sentiment_classes, p0, bert_model, hidden_dim)
    # 定义优化器AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 定义学习率调度器

    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=num_epochs * len(train_dataloader))
    # 定义损失函数的loss返回权重
    sentiment_loss_weight = torch.tensor(sentiment_loss_weight, dtype=torch.float).to(device)
    all_sentiment_loss_weight = torch.tensor(all_sentiment_loss_weight, dtype=torch.float).to(device)
    # 定义损失函数
    criterion_sentiment = CrossEntropyLoss(sentiment_loss_weight)
    criterion_all = CrossEntropyLoss(all_sentiment_loss_weight)
    # 将模型和数据移动到GPU上
    model.to(device)
    criterion_sentiment.to(device)
    criterion_all.to(device)
    # 训练模型
    model.train()

    logging.info("开始训练模型:")

    logging.info(
        "参数:weight:{},p:{},lr:{},累积步数:{},情感loss权重:{},总情感loss权重:{}"
        .format(
            weight,
            p0,
            learning_rate,
            accumulation_steps,
            sentiment_loss_weight,
            all_sentiment_loss_weight
        )
    )
    for epoch in range(num_epochs):

        # 定义进度条
        per = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                   desc="第{}轮训练".format(epoch + 1),
                   ncols=150, position=0, dynamic_ncols=True)

        for i, data in per:
            data = data.to(device)
            # 清空梯度
            if (i + 1) % accumulation_steps == 0:
                optimizer.zero_grad()
            # 前向传播
            entity_loss, entity_output, sentiment_output, all_output = model(data)

            # 计算情感预测判断长度
            if len(sentiment_output) != len(data.aspect_sentiment):
                # 如果预测的长度大于实际的长度，截取
                if len(sentiment_output) > len(data.aspect_sentiment):
                    sentiment_output = sentiment_output[:len(data.aspect_sentiment)]
                else:
                    # 如果预测的长度小于实际的长度，补充使其长度一致
                    pad_tensor = torch.zeros((len(data.aspect_sentiment) -
                                              len(sentiment_output), num_sentiment_classes),
                                             dtype=torch.float)
                    pad_tensor = pad_tensor.to(device)
                    sentiment_output = torch.cat((sentiment_output, pad_tensor), 0)

            # 计算情感预测的loss
            if sentiment_output.shape[0] != 0:  # 如果sentiment_output不为空
                sentiment_loss = criterion_sentiment(sentiment_output, data.aspect_sentiment)
            else:
                if len(data.aspect_sentiment) == 0:
                    sentiment_loss = 0  # 如果sentiment_output为空，且data.y_sentiment为空，那么情感预测的损失为0
                else:
                    sentiment_loss = float('inf')  # 如果sentiment_output为空，那么情感预测的损失为无穷大

            # 计算整体句子情感预测的loss
            all_loss = criterion_all(all_output, data.all_sentiment)
            # 按权重返回总的loss
            loss = weight[0] * entity_loss + weight[1] * sentiment_loss + weight[2] * all_loss

            loss.backward()  # 反向传播
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # 更新模型参数
                scheduler.step()  # 更新学习率
        per.close()

    return model


# 测试模型
# noinspection PyShadowingNames
def test_model(test_dataloader, model, device):
    model.eval()
    pred_entity = []
    pred_sentiment = []
    pred_all = []
    true_entity = []
    true_sentiment = []
    true_all = []
    logging.info("模型测试:")
    with torch.no_grad():
        per = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="测试:", ncols=150,
                   position=0, dynamic_ncols=True)
        for i, data in per:
            data = data.to(device)
            entity_output, sentiment_output, all_output = model(data)
            # 预测实体
            predicted_entity = entity_output
            # 预测实体情感
            _, predicted_sentiment = torch.max(sentiment_output, 1)
            # 预测整体句子情感
            _, predicted_all = torch.max(all_output, 1)
            # 总数统计
            pred_entity.append(predicted_entity)
            pred_sentiment.append(predicted_sentiment.tolist())
            pred_all.append(predicted_all.tolist())
            # 收集实际的实体
            true_entity.append(data.entity)
            true_sentiment.append(data.aspect_sentiment.tolist())
            true_all.append(data.all_sentiment.tolist())
        per.close()

    return pred_entity, pred_sentiment, pred_all, true_entity, true_sentiment, true_all


# noinspection PyShadowingNames
def train_test_model(train_dataloader, tags_size, num_sentiment_classes, all_sentiment_classes,
                     num_epochs, learning_rate, device, weight, p0, accumulation_steps, bert_model, hidden_dim,
                     sentiment_loss_weight, all_sentiment_loss_weight, test_dataloader, model_type):
    # 训练模型
    model = train_model(train_dataloader, tags_size, num_sentiment_classes, all_sentiment_classes,
                        num_epochs, learning_rate, device, weight, p0, accumulation_steps, bert_model, hidden_dim,
                        sentiment_loss_weight, all_sentiment_loss_weight, model_type)

    logging.info("开始测试模型:")

    # 测试模型
    pred_entity, pred_sentiment, pred_all, true_entity, true_sentiment, true_all = test_model(test_dataloader,
                                                                                              model, device)

    return model, pred_entity, pred_sentiment, pred_all, true_entity, true_sentiment, true_all


# noinspection PyShadowingNames


def read_data(catalog, is_bio, bert_path, ltp_path, train_data_path, test_data_path):

    logging.info("读取数据")
    # 读取数据
    train_org_data = read_process_data(train_data_path)
    test_org_data = read_process_data(test_data_path)
    if is_bio:
        logging.info("正在使用BIO标注:")
        # 获取训练数据
        bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_entity_tags(
            train_org_data, catalog, bert_path, ltp_path)
        # 数据转换
        train_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                               True)
        logging.info("训练数据获取完成")
        logging.info("正在使用BIO标注获取测试数据:")
        # 获取测试数据
        bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_entity_tags(
            test_org_data, catalog, bert_path, ltp_path)
        # 数据转换
        test_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                              False)
    else:
        if catalog == 'ltp':
            # 获取训练数据
            logging.info("正在使用ltp获取训练数据:")
            bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_matrix_data_ltp(
                train_org_data, bert_path, ltp_path)
            # 数据转换
            train_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                                   True)
            logging.info("训练数据获取完成")
            logging.info("正在使用ltp获取测试数据:")
            # 获取测试数据
            bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length = get_matrix_data_ltp(
                test_org_data, bert_path, ltp_path)
            # 数据转换
            test_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                                  False)
            logging.info("测试数据获取完成")

        else:
            logging.info("正在使用邻接矩阵获取训练数据:")
            # 获取训练数据
            (bert_input, edge_index, batch, y_entity,
             y_sentiment, all_sentiment, emotion_dict, total_length) = get_matrix_data_adjacency(train_org_data,
                                                                                                 bert_path)
            # 数据转换
            train_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                                   True)
            logging.info("训练数据获取完成")
            logging.info("正在使用邻接矩阵获取测试数据:")
            # 获取测试数据
            (bert_input, edge_index, batch, y_entity,
             y_sentiment, all_sentiment, emotion_dict, total_length) = get_matrix_data_adjacency(test_org_data,
                                                                                                 bert_path)
            # 数据转换
            test_data = MyDataSet(bert_input, edge_index, batch, y_entity, y_sentiment, all_sentiment, total_length,
                                  False)
            logging.info("测试数据获取完成")

    # 构建成dataloader
    train_dataloader = train_data.data_embedding()
    test_dataloader = test_data.data_embedding()
    return train_dataloader, test_dataloader


def evaluate_model(pred_sentiment, pred_all, true_sentiment, true_all):
    # 存储信息 用于评估-评估指标包括：准确率, 精确率, 召回率, F1值, 混淆矩阵
    # 方面级列表[T2,T1,T0,T-1,T-2,F2,F1,F0,F-1,F-2,P2,P1,P0,P-1,P-2,NN,PN]
    # T2: 预测为2,实际为2 T1: 预测为1,实际为1T0: 预测为0，实际为0 T-1: 预测为-1，实际为-1 T-2: 预测为-2，实际为-2
    # F2: 实际为2,预测不为2 F1: 实际为1，预测不为1 F0: 实际为0，预测不为0 F-1: 实际为-1，预测不为-1 F-2: 实际为-2，预测不为-2
    # P2: 预测为2,实际不是2 P1: 预测为1,实际不是1 P0: 预测为0，实际不是0 P-1: 预测为-1，实际不是-1 P-2: 预测为-2，实际不是-2
    # NN: 预测多的 PN: 没有预测的
    # 总情感列表[TP, FP, FN, TN]
    # TP: 预测为正面，实际为正面 FP: 预测为正面，实际为负面  FN: 预测为负面，实际为正面  TN: 预测为负面，实际为负面
    all_num = [0, 0, 0, 0]
    # 计算情感的评估指标
    sentiment_num = [0] * 17
    for i in range(len(pred_sentiment)):
        if len(pred_sentiment[i]) == 0:
            sentiment_num[16] += len(true_sentiment[i])
        else:
            if len(pred_sentiment[i]) != len(true_sentiment[i]):
                if len(pred_sentiment[i]) > len(true_sentiment[i]):
                    sentiment_num[15] += len(pred_sentiment[i]) - len(true_sentiment[i])
                    pred_sentiment[i] = pred_sentiment[i][:len(true_sentiment[i])]
                else:
                    sentiment_num[16] += len(true_sentiment[i]) - len(pred_sentiment[i])
                    true_sentiment[i] = true_sentiment[i][:len(pred_sentiment[i])]
            for j in range(len(pred_sentiment[i])):
                if pred_sentiment[i][j] == true_sentiment[i][j]:
                    sentiment_num[int(true_sentiment[i][j])] += 1
                else:
                    sentiment_num[int(true_sentiment[i][j]) + 5] += 1
                    sentiment_num[int(pred_sentiment[i][j]) + 10] += 1

    # 计算整体句子情感的评估指标
    for i in range(len(pred_all)):
        for j in range(len(pred_all[i])):
            if pred_all[i][j] == true_all[i][j]:
                if pred_all[i][j] == 1:
                    all_num[0] += 1
                else:
                    all_num[3] += 1
            else:
                if pred_all[i][j] == 1:
                    all_num[1] += 1
                else:
                    all_num[2] += 1
    logging.info(
        "情感评估指标:=" + str(sentiment_num))
    logging.info(
        "整体句子情感评估指标:TP:{0},FP:{1},FN:{2},TN:{3}".format(all_num[0], all_num[1], all_num[2], all_num[3]))
    return sentiment_num, all_num


def eval_entity(pred_entity, true_entity, tags_size):
    logging.info("评估模型")
    if tags_size == 3:
        #  计算评估指标,数据格式T0-20(预测和实际一样的),P0-20(实际为，预测不是)，F0-20(预测是，实际不是)
        entity_num = [0, 0, 0, 0]
        # 计算实体的评估指标
        for i in range(len(pred_entity)):
            for j in range(len(pred_entity[i])):
                if pred_entity[i][j] == true_entity[i][j]:
                    if pred_entity[i][j] == 1 or pred_entity[i][j] == 2:
                        entity_num[0] += 1
                    else:
                        entity_num[3] += 1
                else:
                    if pred_entity[i][j] == 1 or pred_entity[i][j] == 2:
                        entity_num[1] += 1
                    else:
                        entity_num[2] += 1
    else:
        # 计算评估指标,数据格式T0-tags_size(预测和实际一样的),P0-tags_size(实际为，预测不是)，F0-tags_size(预测是，实际不是)
        entity_num = [0] * tags_size * 3
        for i in tqdm(range(len(pred_entity)), desc='Evaluate', ncols=100, total=len(true_entity),
                      dynamic_ncols=True):
            for j in range(len(pred_entity[i])):
                if true_entity[i][j] == pred_entity[i][j]:
                    entity_num[true_entity[i][j]] += 1
                else:
                    entity_num[true_entity[i][j] + tags_size] += 1
                    entity_num[pred_entity[i][j] + tags_size * 2] += 1
    logging.info("实体评估指标:{}".format(str(entity_num)))
    return entity_num


if __name__ == '__main__':
    # 忽略 UserWarning 类型的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # 定义BERT模型
    bert_model = BertModel.from_pretrained(bert_path)
    logging.info("BERT模型获取完成")
    # 确定输出的标签种类
    if is_bio:
        # BIO标注
        with open('./process_data/label_dict.json', 'r', encoding='utf-8') as f:
            tags_size = len(json.load(f)) * 2 - 1
            f.close()
    else:
        # 实体的种类+1(0表示不是实体)BIO标注
        tags_size = 3
    # json数据读取
    train_dataloader, test_dataloader = read_data(catalog, is_bio, bert_path, ltp_path, train_data_path, test_data_path)
    logging.info("数据读取以及转换完成")
    logging.info("开始训练测试模型：")
    # 训练测试模型
    all_results = []
    data = []
    for weight in weights:
        model, pred_entity, pred_sentiment, pred_all, true_entity, true_sentiment, true_all = train_test_model(
            train_dataloader, tags_size, num_sentiment_classes, all_sentiment_classes, num_epochs, learning_rate,
            device, weight, p0, accumulation_steps, bert_model, hidden_dim, sentiment_loss_weight,
            all_sentiment_loss_weight, test_dataloader, model_type)
        data.append({'model_type': model_type, 'catalog': catalog, 'weight': weight,
                     'entity': [pred_entity, true_entity], 'sentiment': [pred_sentiment, true_sentiment],
                     'all': [pred_all, true_all]})
        # 把data保存到文件
        with open('./result/data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)
            f.close()
        # 评估模型
        entity_num = eval_entity(pred_entity, true_entity, tags_size)
        sentiment_num, all_num = evaluate_model(pred_sentiment,
                                                pred_all, true_sentiment, true_all)

        all_results.append({'model_type': model_type, 'catalog': catalog, 'epoch': num_epochs,
                            'learning_rate': learning_rate, 'p': p0,'accumulation_steps': accumulation_steps,
                            'weight': weight, 'entity_num': entity_num, 'sentiment_num': sentiment_num,
                            'all_num': all_num})
        # 保存模型和预测结果
        with open('./result/all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f)
            f.close()
        logging.info(
            "参数:weight:{},p:{},lr:{},累积步数:{},情感loss权重:{},总情感loss权重:{}的模型训练完成:"
            .format(
                weights,
                p0,
                learning_rate,
                accumulation_steps,
                sentiment_loss_weight,
                all_sentiment_loss_weight
            )
        )
