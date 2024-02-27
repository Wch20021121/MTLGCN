import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from MTLGCN import MultitaskGCN


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    entity_output, sentiment_output = model(data)
    entity_labels, sentiment_labels = data.y_entity, data.y_sentiment

    loss1 = criterion(entity_output, entity_labels)
    loss2 = criterion(sentiment_output, sentiment_labels)
    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    return loss.item()


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def predict(model, data):
    model.eval()
    with torch.no_grad():
        entity_output, sentiment_output = model(data)
    return entity_output, sentiment_output


# 创建模型
num_node_features = 1
num_entity_classes = 2
num_sentiment_classes = 5
model = MultitaskGCN(num_node_features, num_entity_classes, num_sentiment_classes)

# 创建优化器和损失函数
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
data = []
new_data = []
# 训练模型
for epoch in range(3):
    loss = train(model, data, optimizer, criterion)
    print(f'Epoch: {epoch + 1}, Loss: {loss}')

# 保存模型
save_model(model, 'model.pth')

# 加载模型
model.load_state_dict(torch.load('model.pth'))

# 预测新的数据
entity_output, sentiment_output = predict(model, new_data)
