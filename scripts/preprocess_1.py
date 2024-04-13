# Define the class to encode dataset and function to slpit the dataset in train, val and test
# plus define function to encode training data

class ABSA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def merge_sent(list1, list2):
    merged_ = []
    for x, y in zip(list1, list2):
        merged_.append(x + " [SEP] " + str(y))
    return merged_

def split_dataframe(df, train_percent=0.7, val_percent=0.15, test_percent=0.15):

    if train_percent + val_percent + test_percent != 1.0:
        raise ValueError("The sum of train_percent, val_percent, and test_percent must be equal to 1.0.")

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split indices
    total_samples = len(df)
    train_end = int(train_percent * total_samples)
    val_end = train_end + int(val_percent * total_samples)

    # Split the DataFrame
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def setup_data(tokenizer, train_df, eval_df, test_df):

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'conflict': 3}
    label_map_category = {'service': 4, 'food': 5,'anecdotes/miscellaneous': 6,'price': 7, 'ambience': 8 }

    train_texts = train_df['sentence']
    train_labels = train_df['polarity']
    train_labels_category = train_df['category']
    train_labels = [label_map[label_] for label_ in train_labels]
    train_labels_category = [label_map_category[label] for label in train_labels_category]


    eval_texts = eval_df['sentence']
    eval_labels = eval_df['polarity']
    eval_labels_category = eval_df['category']
    eval_labels = [label_map[label_] for label_ in eval_labels]
    eval_labels_category = [label_map_category[label] for label in eval_labels_category]

    test_texts = test_df['sentence']
    test_labels = test_df['polarity']
    test_labels_category = test_df['category']
    test_labels = [label_map[label_] for label_ in test_labels]
    test_labels_category = [label_map_category[label] for label in test_labels_category]

    # Since DeBERTa does not accept 2 labels as input, I'm including the categories in the text embeddings
    # separated by a special tokens [SEP]
    train_ = merge_sent(train_texts, train_labels_category)
    eval_ = merge_sent(eval_texts, eval_labels_category)
    test_ = merge_sent(test_texts, test_labels_category)

    # train_encodings = tokenizer(str(train_texts), padding="max_length", truncation=True, max_length=512)
    # eval_encodings = tokenizer(str(eval_texts), padding="max_length", truncation=True, max_length=512)
    # test_encodings = tokenizer(str(test_texts), padding="max_length", truncation=True, max_length=512)

    train_encodings = tokenizer(train_, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_, padding="max_length", truncation=True, max_length=512)
    test_encodings = tokenizer(test_, padding="max_length", truncation=True, max_length=512)

    #train_labels = [train_labels, train_labels_category]
    #eval_labels = [eval_labels, eval_labels_category]

    train_dataset = ABSA_Dataset(train_encodings, train_labels)
    val_dataset = ABSA_Dataset(eval_encodings, eval_labels)

    return train_dataset, val_dataset, test_
