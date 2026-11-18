import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "hello how are you",
    "i am fine thank you",
    "how are you doing",
    "i am doing well",
    "i am doing good"
]

# 2. Sözlük (Vocabulary) Oluşturma
words = sorted(list(set(" ".join(sentences).split())))
word2idx = {w: i for i, w in enumerate(words)}
indexToWord = {i: w for i, w in enumerate(words)}  
vocabSize = len(words)

seq_len = 2  

data_X, data_Y = [], []

for sentence in sentences:
    tokens = sentence.split()
    for i in range(len(tokens) - seq_len):
        inputSeq = [word2idx[w] for w in tokens[i:i + seq_len]]
        targetWord = word2idx[tokens[i + seq_len]]
        data_X.append(inputSeq)
        data_Y.append(targetWord)

x_train = torch.tensor(data_X, dtype=torch.long)
y_train = torch.tensor(data_Y, dtype=torch.long)


class SimpleChatModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        
        lstm_out, _ = self.lstm(embeds)
        
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # Crucial step: The LSTM processes the entire sequence, but to predict the 
        # next word, we only care about the accumulated context at the LAST time step (-1).
        # For example, given ["how", "are"], we take the state after "are" to predict "you".
        # Slicing [:, -1, :] extracts this final hidden state and feeds it into the fc layer.
        out = self.fc(lstm_out[:, -1, :])
        return out


model = SimpleChatModel(vocabSize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()


def nextWord(prompt):
    model.eval()
    tokens = prompt.lower().split()[-seq_len:]
    
    for w in tokens:
        if w not in word2idx:
            return f"'{w}' kelimesi sözlükte bulunamadı!"
            
    input_indices = torch.tensor([[word2idx[w] for w in tokens]], dtype=torch.long)
    with torch.no_grad():
        prediction = model(input_indices)
        prediction_index = torch.argmax(prediction, dim=1).item()
        
    return indexToWord[prediction_index]  


# Test
print("Girdi: 'how are' -> Tahmin:", nextWord("how are"))
print("Girdi: 'i am'    -> Tahmin:", nextWord("i am"))
print("Girdi: 'i am doing '    -> Tahmin:", nextWord(nextWord("i am")))
print("Girdi: 'i am doing '    -> Tahmin:", nextWord(nextWord("i am")))
print("Girdi: 'i am doing '    -> Tahmin:", nextWord(nextWord("i am")))
print("Girdi: 'i am doing '    -> Tahmin:", nextWord(nextWord("i am"))) 
# it only says good, didn't said well not even once 
# ok now only says well 