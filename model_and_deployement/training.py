vocab = {"<PAD>": 0, "<UNK>": 1}
max_length = 40
lemmatizer = WordNetLemmatizer()
for sentence in x_data:
  tokens = word_tokenize(sentence)
  for word in tokens:
    new_word = lemmatizer.lemmatize(word.lower())
    if new_word not in vocab:
        vocab[new_word] = len(vocab)

sequences = []
for sentence in x_data:
    tokens = word_tokenize(sentence)
    seq = []
    for word in tokens:
        new_word = lemmatizer.lemmatize(word.lower())
        seq.append(vocab.get(new_word, vocab["<UNK>"]))
    seq = seq[:max_length] + [vocab["<PAD>"]] * (max_length - len(seq))
    sequences.append(seq)

X_train, X_test, y_train, y_test = train_test_split(sequences, y_data, test_size=0.2)

print(len(X_train))
print(len(X_test))
print(X_test)
print("Sample X_train:", X_train[0])

X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train).to(device)
X_test = torch.tensor(X_test, dtype=torch.long).to(device)
y_test = torch.tensor(y_test).to(device)

class TonalityLSTM(nn.Module):
    def __init__(self, vocab_shape, embedding_shape, middle_shape, output_shape):
        super().__init__()
        self.embedding = nn.Embedding(vocab_shape, embedding_shape)
        self.lstm = nn.LSTM(embedding_shape, middle_shape, batch_first=True)
        self.fc = nn.Linear(middle_shape, output_shape)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

vocab_length = len(vocab)
embedding_length = 100
hidden_length = 64
classifications = 3

model = TonalityLSTM(vocab_length, embedding_length, hidden_length, classifications).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
losses = []
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

with torch.no_grad():
    test_preds = model(X_test)
    probabilities = torch.softmax(test_preds, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    accuracy = (predicted_labels == y_test).float().mean()
    print(f"Test Accuracy: {accuracy}")
