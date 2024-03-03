import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from MTLGCN import MultitaskGCN
import json
from DataSets import MyDataSet
import random


# noinspection PyShadowingNames
def train_model(train_data, num_node_features, num_entity_classes, num_sentiment_classes, all_sentiment_classes,
                num_epochs, learning_rate, device, weights):
    # 初始化模型和优化器
    model = MultitaskGCN(num_node_features, num_entity_classes, num_sentiment_classes, all_sentiment_classes)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    # 将模型和数据移动到GPU上
    model.to(device)
    criterion.to(device)

    # 训练模型
    model.train()
    # 训练循环
    print("=====================================开始训练=====================================")
    for epoch in range(num_epochs):
        for data in train_data:
            data = data.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            entity_output, sentiment_output, all_output, entity_mask = model(data)

            # 计算实体预测的loss
            entity_loss = criterion(entity_output, data.y_entity)

            # 计算情感预测的loss
            sentiment_loss = criterion(sentiment_output, data.y_sentiment[entity_mask])

            # 计算整体句子情感预测的loss
            all_loss = criterion(all_output, data.all_sentiment)
            # 按权重返回总的loss
            loss = weights[0] * entity_loss + weights[1] * sentiment_loss + weights[2] * all_loss

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
        print("第{}轮训练完成".format(epoch + 1))

    return model


def convert_to_list(predictions):
    result = []
    for prediction in predictions:
        result.append(prediction[0].tolist())
    return result


# 测试模型
# noinspection PyShadowingNames
def test_model(test_data, model, device):
    model.eval()
    correct_entity = 0
    correct_sentiment = 0
    correct_all = 0
    total_entity = 0
    total_sentiment = 0
    total_all = 0
    pre_entity = []
    pre_sentiment = []
    pre_all = []
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            entity_output, sentiment_output, all_output, entity_mask = model(data)
            _, predicted_entity = torch.max(entity_output, 1)
            _, predicted_sentiment = torch.max(sentiment_output, 1)
            _, predicted_all = torch.max(all_output, 1)
            # 总数统计
            total_entity += data.y_entity.size(0)
            total_sentiment += data.y_sentiment[entity_mask].size(0)
            total_all += data.all_sentiment.size(0)
            pre_entity.append([predicted_entity, data.y_entity])
            pre_sentiment.append([predicted_sentiment, data.y_sentiment[entity_mask]])
            pre_all.append([predicted_all, data.all_sentiment])
            correct_entity += (predicted_entity == data.y_entity).sum().item()
            correct_sentiment += (predicted_sentiment == data.y_sentiment[entity_mask]).sum().item()
            correct_all += (predicted_all == data.all_sentiment).sum().item()

    print('Accuracy of the network on the entity: %d %%' % (100 * correct_entity / total_entity))
    if total_sentiment == 0:
        print('没有预测一个是实体，全部预测是非实体')
    else:
        print('Accuracy of the network on the sentiment: %d %%' % (100 * correct_sentiment / total_sentiment))
    print('Accuracy of the network on the all: %d %%' % (100 * correct_all / total_all))
    pre_entity = convert_to_list(pre_entity)
    pre_sentiment = convert_to_list(pre_sentiment)
    pre_all = convert_to_list(pre_all)
    return pre_entity, pre_sentiment, pre_all


# noinspection PyShadowingNames
def train_test_model(train_data, num_node_features, num_entity_classes, num_sentiment_classes, all_sentiment_classes,
                     num_epochs, learning_rate, device, weights):
    # 训练模型
    model = train_model(train_data, num_node_features, num_entity_classes, num_sentiment_classes, all_sentiment_classes,
                        num_epochs, learning_rate, device, weights)
    print("================================模型训练完成================================")
    # 测试模型
    print("================================开始测试模型================================")
    pre_entity, pre_sentiment, pre_all = test_model(test_data, model, device)
    print("================================模型测试完成================================")
    return model, pre_entity, pre_sentiment, pre_all


# noinspection PyShadowingNames
def save_data(model, pre_entity, pre_sentiment, pre_all, path, name):
    path_name = path + '/' + name + '.pkl'
    # 保存模型
    torch.save(model, path_name)
    # 保存预测结果做为json保存
    with open(path + '/' + 'pre_entity.json', 'w') as f:
        json.dump(pre_entity, f)
        f.close()
    with open(path + '/' + 'pre_sentiment.json', 'w') as f:
        json.dump(pre_sentiment, f)
        f.close()
    with open(path + '/' + 'pre_all.json', 'w') as f:
        json.dump(pre_all, f)
        f.close()


def read_data(path):
    with open(path + '/' + 'all_sentiment.json', 'r') as f:
        all_sentiment = json.load(f)
        f.close()
    with open(path + '/' + 'batch.json', 'r') as f:
        batch = json.load(f)
        f.close()
    with open(path + '/' + 'edge_index.json', 'r') as f:
        edge_index = json.load(f)
        f.close()
    with open(path + '/' + 'node_feature.json', 'r') as f:
        node_feature = json.load(f)
        f.close()
    with open(path + '/' + 'y_entity.json', 'r') as f:
        y_entity = json.load(f)
        f.close()
    with open(path + '/' + 'y_sentiment.json', 'r') as f:
        y_sentiment = json.load(f)
        f.close()
    return all_sentiment, batch, edge_index, node_feature, y_entity, y_sentiment


if __name__ == '__main__':
    # 定义超参数
    num_epochs = 3
    learning_rate = 0.05
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型的输入维度和输出维度
    num_node_features = 1
    num_entity_classes = 2
    num_sentiment_classes = 6
    all_sentiment_classes = 2
    # 定义权重
    weights = [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    print("================================超参数定义完成================================")
    # json数据读取
    all_sentiment, batch, edge_index, node_feature, y_entity, y_sentiment = read_data('./process_data/A_use_data')
    print("================================数据读取完成================================")
    # 数据转换
    MyDataSet = MyDataSet(node_feature, edge_index, batch, y_entity, y_sentiment, all_sentiment)
    datalist = MyDataSet.data_embedding()
    print("================================数据转换完成================================")
    # 数据集划分打乱数据集
    # 打乱数据
    random.shuffle(datalist)

    # 计算训练集的大小
    train_size = int(len(datalist) * 0.7)

    # 切分数据
    train_data = datalist[0:train_size]
    test_data = datalist[train_size:]
    print("================================数据集划分完成================================")
    # 训练测试模型
    for weight in weights:
        model, pre_entity, pre_sentiment, pre_all = train_test_model(train_data, num_node_features, num_entity_classes,
                                                                     num_sentiment_classes, all_sentiment_classes,
                                                                     num_epochs,
                                                                     learning_rate, device, weight)
