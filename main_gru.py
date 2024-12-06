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

# Гиперпараметры
batch_size = 256        # Размер батча
seq_length = 15         # Длина последовательности для обучения
hidden_size = 400       # Размер скрытого слоя GRU
num_layers = 4          # Количество слоёв GRU
embed_size = 1024       # Размер слоя эмбеддингов
learning_rate = 0.0005  # Начальная скорость обучения

# Режим программы: "compress", "decompress", или "both"
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
        # Предсказываем логиты следующего символа
        logits = self.fc(h_n)  # (batch_size, vocab_size)
        return logits


def get_symbol(index, length, freq, coder, compress, data):
    """Возвращает следующий символ с использованием арифметического кодирования."""
    symbol = 0
    if index < length:
        if compress:
            # При сжатии записываем символ в кодировщик
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            # При декомпрессии считываем символ из декодера
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


# Обучающий шаг
def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data):
    """Выполняет одну итерацию обучения или декодирования."""
    loss = 0
    cross_entropy = 0
    denom = 0
    split = math.ceil(length / batch_size)

    model.train()
    optimizer.zero_grad()

    seq_input = seq_input.to(device)

    logits = model(seq_input)  # Логиты (batch_size, vocab_size)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # Преобразуем в вероятности

    symbols = []  # Хранение таргетных символов
    mask = []     # Маска для обработки случая, когда батч может быть неполностью заполнен

    # Предсказываем символы
    for i in range(batch_size):
        freq = np.cumsum(probs[i] * 10000000 + 1)  # Частоты для арифметического кодера
        index = pos + i * split
        symbol = get_symbol(index, length, freq, coder, compress, data)
        symbols.append(symbol)

        if index < length:
            prob = probs[i][symbol]  # Вероятность символа
            if prob <= 0:
                prob = 1e-6  # Избегаем log(0)
            cross_entropy += math.log2(prob)  # Суммируем кросс-энтропию
            denom += 1
            mask.append(1.0)
        else:
            mask.append(0.0)

    # Преобразуем таргетные символы в one-hot
    symbols = torch.tensor(symbols, device=device)
    input_one_hot = torch.nn.functional.one_hot(symbols, vocab_size).float()

    # Лосс
    loss = torch.nn.functional.cross_entropy(logits, input_one_hot, reduction='none')
    loss = loss * torch.tensor(mask, device=device).unsqueeze(1)  # Применяем маску
    loss = loss.mean()  # Среднее значение лосса

    # Обратное распространение
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 4)  # Ограничение градиентов
    optimizer.step()

    # Обновляем входную последовательность
    seq_input = torch.cat([seq_input[:, 1:], symbols.unsqueeze(1)], dim=1)

    return seq_input, cross_entropy, denom

# Основной процесс сжатия/декомпрессии
def process(compress, length, vocab_size, coder, data):
    """Выполняет сжатие или декомпрессию."""
    start = time.time()
    
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    # Модель
    model = GRUCompress(vocab_size, embed_size, hidden_size, num_layers).to(device)
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Задаем первичное равномерное распределение вероятностей для автокодировщика для начальных символов
    freq = np.cumsum(np.full(vocab_size, 1.0 / vocab_size) * 10000000 + 1)
    symbols = []
    for i in range(batch_size):
        symbols.append(get_symbol(i * (length // batch_size), length, freq, coder, compress, data))

    # Начальные токены копируем на всю последовательность
    seq_input = torch.tensor(symbols, device=device).unsqueeze(1).repeat(1, seq_length)

    pos = 0 # Номер текущего батча
    cross_entropy = 0
    denom = 0 # Счётчик количества символов, для которых была рассчитана кросс-энтропия

    split = math.ceil(length / batch_size)  # Кол-во батчей
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'

    # Основной цикл обучения
    while pos < split:
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
    """Сжимает данные из исходного файла и сохраняет результат в файл .dat."""
    int_list = [] # Список для хранения символов из файла
    text = open(path_to_file, 'rb').read()  # Читаем файл в бинарном формате
    vocab = sorted(set(text))  # Формируем уникальный словарь символов
    vocab_size = len(vocab)  # Размер словаря

    # Создаём отображение символ -> индекс
    char2idx = {u: i for i, u in enumerate(vocab)}
    for _, c in enumerate(text):
        int_list.append(char2idx[c])  # Преобразуем каждый символ в его индекс

    # Округляем размер словаря до ближайшего числа, кратного 8, для оптимизации
    vocab_size = math.ceil(vocab_size / 8) * 8
    file_len = len(int_list)  # Количество символов в файле
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        # Записываем длину исходного файла в заголовок сжатого файла
        length = len(int_list)
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        
        # Записываем информацию о словаре в заголовок файла (256 бит для каждого символа)
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)

        # Инициализируем арифметический кодировщик
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)  # Запускаем процесс сжатия


def decompression():
    """Декомпрессирует данные из сжатого файла и сохраняет результат в новый файл."""
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        # Считываем длину исходного файла из заголовка
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)  # Пропускаем байты, содержащие длину

        # Инициализируем список для хранения декодированных символов
        output = [0] * length
        bitin = BitInputStream(inp)  # Поток для чтения побитово

        # Считываем словарь из заголовка файла
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        
        # Округляем размер словаря до ближайшего числа, кратного 8
        vocab_size = math.ceil(vocab_size / 8) * 8
        
        # Инициализируем арифметический декодер
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)  # Запускаем процесс декомпрессии

        # Преобразуем индексы обратно в символы и записываем в выходной файл
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    """Основная функция, управляющая процессами сжатия и декомпрессии."""
    start = time.time()
    if mode == 'compress' or mode == 'both':
        # Выполняем сжатие
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file) / os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        # Выполняем декомпрессию
        decompression()
        # Проверяем совпадение хэшей исходного и восстановленного файлов
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig, "Восстановленный файл не совпадает с оригиналом!"
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()