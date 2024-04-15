class ABSA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([label[idx] for label in self.labels])  # Assuming self.labels is a list of lists
        return item

    def __len__(self):
        return len(self.labels[0])  # Assuming all labels have the same length

def parse_data_2014(xml_file):
    container = []  # Initialize Container (List) for Parse Data
    sentences = ET.parse(xml_file).getroot()  # Get Sentence-Level Nodes

    for sentence in sentences:  # Loop Through Sentences
        sentence_id = sentence.attrib["id"]  # Save ID
        sentence_text = sentence.find('text').text  # Save Text
        aspects = sentence.findall('*')  # Get Aspect-Level Nodes

        found_category = False

        for aspect in aspects:  # Loop Through Aspects
            if aspect.tag == "aspectCategories":
                opinions = aspect.findall('*')  # Get Opinion-Level Nodes
                for opinion in opinions:
                    category = opinion.attrib["category"]
                    polarity = opinion.attrib.get("polarity", np.nan)
                    row = {"sentence_id": sentence_id, "sentence": sentence_text, "category": category, "polarity": polarity}
                    container.append(row)
                found_category = True

        if not found_category:
            row = {"sentence_id": sentence_id, "sentence": sentence_text, "category": np.nan, "polarity": np.nan}
            container.append(row)

    return pd.DataFrame(container)


# resulting DF
df = rests
todrop=['sentence_id']
df.drop(columns=todrop, inplace=True)
df.head(10)


rests = parse_data_2014("/kaggle/input/restaurants-train/Restaurants_Train.xml")


def setup_data(parsed_data, tokenizer, train_split, eval_split):

    df = parsed_data
    df = Dataset.from_pandas(df)

    train_df = df.select([i for i in range(train_split)])
    eval_df = df.select([i for i in range(train_split, eval_split)])

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'conflict': 3}
    label_map_category = {'service': 0, 'food': 1, 'anecdotes/miscellaneous': 2, 'price': 3, 'ambience': 4}

    train_texts = train_df['sentence']
    train_labels = train_df['polarity']
    train_labels_category = train_df['category']  # Assuming 'category' column is the label
    train_labels = [[label_map[label] for label in train_labels],  # Convert labels to numerical IDs
                    [label_map_category[label] for label in train_labels_category]]  # Convert category labels to numerical IDs

    eval_texts = eval_df['sentence']
    eval_labels = eval_df['polarity']
    eval_labels_category = eval_df['category']
    eval_labels = [[label_map[label] for label in eval_labels],
                   [label_map_category[label] for label in eval_labels_category]]

    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=512)

    train_dataset = ABSA_Dataset(train_encodings, train_labels)
    val_dataset = ABSA_Dataset(eval_encodings, eval_labels)

    return train_dataset, val_dataset
