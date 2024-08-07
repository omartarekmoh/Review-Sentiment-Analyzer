import numpy as np
import pandas as pd
from collections import Counter
import string
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx == None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.add_unk = add_unk
        self.unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(self.unk_token)

    def add_token(self, token):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        index = len(self.token_to_idx)
        self.token_to_idx[token] = index
        self.idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        if self.add_unk:
            return self.token_to_idx.get(token, self.unk_index)

        return self.token_to_idx[token]

    def lookup_index(self, idx):
        if idx in self.idx_to_token:
            return self.idx_to_token[idx]

        raise KeyError(f"The index ({idx}) isn't in the Vocabulary")

    def __str__(self):
        return f"Vocabulary size: {len(self.token_to_idx)}"

    def __len__(self):
        return len(self.token_to_idx)

    def to_serializable(self):
        return {
            "token_to_idx": self.token_to_idx,
            "add_unk": self.add_unk,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)


class ReviewVectorizer:
    def __init__(self, review_vocab, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for word in review.split():
            if word not in string.punctuation:
                idx = self.review_vocab.lookup_token(word)
                one_hot[idx] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, dataframe, cut_off=25):
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        for rating in sorted(dataframe["ratings"].unique()):
            rating_vocab.add_token(rating)

        counter = Counter()
        for _, row in dataframe.iterrows():
            for word in row.reviews.split():
                if word not in string.punctuation:
                    counter[word] += 1

        for word, count in counter.items():
            if count >= cut_off:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    def to_serializable(self):
        return {
            "review_vocab": self.review_vocab.to_serializable(),
            "rating_vocab": self.rating_vocab.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, content):
        review_vocab = Vocabulary.from_serializable(content["review_vocab"])
        rating_vocab = Vocabulary.from_serializable(content["rating_vocab"])

        return cls(review_vocab, rating_vocab)


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        self.vectorizer = vectorizer
        self.review_df = review_df

        self.train_df = self.review_df[self.review_df["split"] == "train"]
        self.train_size = len(self.train_df)
        self.val_df = self.review_df[self.review_df["split"] == "val"]
        self.val_size = len(self.val_df)
        self.test_df = self.review_df[self.review_df["split"] == "test"]
        self.test_size = len(self.test_df)

        self.lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_json):
        review_df = pd.read_csv(review_csv)
        vectorizer = ReviewVectorizer.from_serializable(vectorizer_json)
        return cls(review_df, vectorizer)

    def set_split(self, split):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def get_vectorizer(self):
        return self.vectorizer

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        row = self.target_df.iloc[idx]
        review_vector = self.vectorizer.vectorize(row.reviews)
        rating_index = self.vectorizer.rating_vocab.lookup_token(row.ratings)

        return {"x_data": review_vector, "y_target": rating_index}
