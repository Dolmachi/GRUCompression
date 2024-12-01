import torch
from torch import nn
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The batch size for training
batch_size = 256
# The sequence length for training
seq_length = 15
# The number of units in each GRU layer
hidden_size = 400
# The number of GRU layers
num_layers = 4
# The size of the embedding layer
embed_size = 1024
# The initial learning rate for optimizer
learning_rate = 0.0005

# The mode for the program, "compress", "decompress", "both"
mode = 'both'

path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


class GRUCompress(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(num_layers * hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        _, h_n = self.gru(embeds)  # h_n: (num_layers, batch_size, hidden_size)
        h_n = h_n.transpose(0, 1).reshape(batch_size, num_layers * hidden_size)  # (batch_size, num_layers * hidden_size)
        logits = self.fc(h_n)  # (batch_size, vocab_size)
        return logits
    

def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the *first* batch.
        seq_input: Tensor, containing the last seq_length inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        seq_input: Tensor, containing the last seq_length inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    loss = 0
    cross_entropy = 0
    denom = 0
    split = math.ceil(length / batch_size)

    model.train()  # Устанавливаем режим тренировки
    optimizer.zero_grad()  # Обнуляем градиенты

    seq_input = seq_input.to(device)  # Перевод на GPU/CPU

    # Forward pass
    logits = model(seq_input)  # Получаем логиты (batch_size, vocab_size)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    symbols = []
    mask = []

    # Актуализируем вероятности и маски
    for i in range(batch_size):
        freq = np.cumsum(probs[i] * 10000000 + 1)
        index = pos + i * split
        symbol = get_symbol(index, length, freq, coder, compress, data)
        symbols.append(symbol)

        if index < length:
            prob = probs[i][symbol]
            if prob <= 0:
                prob = 1e-6  # Избегаем ошибки с log
            cross_entropy += math.log2(prob)
            denom += 1
            mask.append(1.0)
        else:
            mask.append(0.0)

    # Преобразование символов в one-hot вектор
    symbols = torch.tensor(symbols, device=device)
    input_one_hot = torch.nn.functional.one_hot(symbols, vocab_size).float()

    # Loss calculation
    loss = torch.nn.functional.cross_entropy(logits, input_one_hot, reduction='none')
    loss = loss * torch.tensor(mask, device=device).unsqueeze(1)  # Применяем маску
    loss = loss.mean()  # Среднее значение лосса

    # Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 4)  # Ограничиваем градиенты
    optimizer.step()

    # Обновляем входную последовательность
    seq_input = torch.cat([seq_input[:, 1:], symbols.unsqueeze(1)], dim=1)

    return seq_input, cross_entropy, denom


def process(compress, length, vocab_size, coder, data):
    """Runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()
    
    # Создание модели
    model = GRUCompress(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.eval()  # Устанавливаем режим оценки

    # Инициализация оптимизатора и лосса
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Подготовка первого батча символов
    freq = np.cumsum(np.full(vocab_size, 1.0 / vocab_size) * 10000000 + 1)
    symbols = []
    for i in range(batch_size):
        symbols.append(get_symbol(i * (length // batch_size), length, freq, coder, compress, data))
    
    seq_input = torch.tensor(symbols, device=device).unsqueeze(1).repeat(1, seq_length)

    pos = 0
    cross_entropy = 0
    denom = 0

    split = math.ceil(length / batch_size)
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'
    
    while pos < split:
        # Тренировочный/инференс шаг
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data)
        cross_entropy += ce
        denom += d
        pos += 1

        if pos % 5 == 0:
            percentage = 100 * pos / split
            print(template.format(percentage, -cross_entropy / denom, time.time() - start))
    
    if compress:
        coder.finish()
    
    print(template.format(100, -cross_entropy / length, time.time() - start))
    
    
def compession():
    # int_list will contain the characters of the file.
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}
    for _ , c in enumerate(text):
        int_list.append(char2idx[c])

    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size/8) * 8
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        # Write 256 bits to the compressed file header to keep track of the vocabulary.
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)
        
        
def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        # Read the original file size from the header.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        # Create a list to store the file characters.
        output = [0] * length
        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        # Round up to a multiple of 8 to improve performance.
        vocab_size = math.ceil(vocab_size/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        # The decompressed data is stored in the "output" list. We can now write the
        # data to file (based on the type of preprocessing used).

        # Convert indexes back to the original characters.
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))
            

def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file)/os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)
    
    
if __name__ == '__main__':
    main()
