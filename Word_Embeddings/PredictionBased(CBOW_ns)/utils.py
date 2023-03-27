import re


class Tokenizer:
    def __init__(self):
        pass

    def clean(self, line):
        return re.sub(r'[^a-z]', ' ', line.lower())

    def tokenize(self, line):
        line = self.clean(line)
        return line.split()


class Review:
    tokenizer = Tokenizer()

    def __init__(self, r):
        self.text = f"{r['reviewText']} {r['summary']}"
        self.tokens = Review.tokenizer.tokenize(self.text)

    def __iter__(self):
        return iter(self.tokens)

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]
