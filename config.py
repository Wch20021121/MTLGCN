import torch

# 定义模型的超参数
num_epochs = 3
learning_rate = 3e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义模型的输入维度和输出维度
# 实体情感的种类
num_sentiment_classes = 5
# 整体句子的情感种类
all_sentiment_classes = 2
p0 = 0.1
# 累计梯度步数
accumulation_steps = 2
# 定义实体情感的loss权重
sentiment_loss_weight = [1, 1, 1, 1, 1]
# 定义整体句子情感的loss权重
all_sentiment_loss_weight = [1, 1]
# 定义模型loss权重
weights = [[1.0, 1.0, 1.0], [1.5, 0.9, 0.6], [0.9, 0.6, 1.5], [0.6, 1.5, 0.9]]
# 数据种类定义 ltp:使用ltp获取边链接关系  adj:使用邻接矩阵获取边链接关系
catalog = 'ltp'
# 是否使用bio标注
is_bio = True
# 预训练好的bert模型路径
bert_path = r'D:\Model\transformers\text-classier\bert_model\bert-base-chinese'
# 定义模型的输出维度
hidden_dim = 256
# 模型采用什么GCN还是GAT
model_type = 'GAT'
# ltp模型路径
ltp_path = './LTP_model'
# 训练数据在process_data里的路径
train_data_path = 'all_spark_train_data.json'  # 所以的al创建的数据和原数据合在一起
# 测试数据名称
test_data_path = 'test_data.json'
