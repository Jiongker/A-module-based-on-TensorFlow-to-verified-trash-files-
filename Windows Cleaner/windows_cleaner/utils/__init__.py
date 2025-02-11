import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(data_dir):
    """
    从指定的数据目录加载文件路径和对应的标签。

    参数:
    data_dir (str): 数据目录的路径，该目录应包含 junk_files、normal_essential 和 critical_essential 子目录。

    返回:
    list: 文件路径列表。
    list: 对应的标签列表，0 表示垃圾文件，1 表示正常不可删除文件，2 表示重要不可删除文件。
    """
    file_paths = []
    labels = []
    junk_dir = os.path.join(data_dir, 'junk_files')
    normal_essential_dir = os.path.join(data_dir, 'normal_essential')
    critical_essential_dir = os.path.join(data_dir, 'critical_essential')

    for file in os.listdir(junk_dir):
        file_paths.append(os.path.join(junk_dir, file))
        labels.append(0)

    for file in os.listdir(normal_essential_dir):
        file_paths.append(os.path.join(normal_essential_dir, file))
        labels.append(1)

    for file in os.listdir(critical_essential_dir):
        file_paths.append(os.path.join(critical_essential_dir, file))
        labels.append(2)

    return file_paths, labels

def preprocess_data(file_paths, labels, max_words=10000, max_length=200):
    """
    对文件数据进行预处理，包括读取文件内容、提取元数据、分词和归一化等操作。

    参数:
    file_paths (list): 文件路径列表。
    labels (list): 对应的标签列表。
    max_words (int): 最大词汇量。
    max_length (int): 序列最大长度。

    返回:
    np.ndarray: 处理后的特征矩阵。
    np.ndarray: 标签数组。
    Tokenizer: 分词器对象。
    MinMaxScaler: 归一化器对象。
    """
    texts = []
    metadata = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            creation_time = file_stats.st_ctime
            modification_time = file_stats.st_mtime
            metadata.append([file_size, creation_time, modification_time])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_text = pad_sequences(sequences, maxlen=max_length)

    metadata = np.array(metadata)
    scaler = MinMaxScaler()
    X_metadata = scaler.fit_transform(metadata)

    X = np.hstack((X_text, X_metadata))
    y = np.array(labels)

    return X, y, tokenizer, scaler

