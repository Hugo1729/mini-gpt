import numpy as np

#excerpt from Hamlet act 1 scene 1
text = """Enter Barnardo and Francisco, two sentinels Meeting

BARNARDO Who's there?

FRANCISCO Nay, answer me: stand and unfold yourself.

BARNARDO Long live the king!

FRANCISCO Barnardo?

BARNARDO He.

FRANCISCO You come most carefully upon your hour.

BARNARDO 'Tis now struck twelve: get thee to bed, Francisco.

FRANCISCO For this relief much thanks: 'tis bitter cold,
And I am sick at heart.

BARNARDO Have you had quiet guard?

FRANCISCO Not a mouse stirring.

BARNARDO Well, goodnight.
If you do meet Horatio and Marcellus,
The rivals of my watch, bid them make haste."""


def get_cnts(data):
    ans = {}

    for x, y in zip(data[:-1], data[1:]):
        ans[(x,y)] = ans.get((x,y), 0) + 1

    return ans

def merge(data, token1, token2, token3):
    ans = []

    i = 0

    while i < len(data):
        if (i+1 < len(data) and data[i] == token1 and data[i+1] == token2):
            ans.append(token3)
            i += 2
        else:
            ans.append(data[i])
            i += 1

    return ans

class BPE:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def train(self, text, num_tokens):
        data = text.encode("utf-8")

        while (len(self.vocab) < num_tokens):
            cnts = get_cnts(data)

            token1, token2 = max(cnts, key = cnts.get)

            token3 = len(self.vocab)

            data = merge(data, token1, token2, token3)

            self.vocab[token3] = self.vocab[token1] + self.vocab[token2]
            self.merges[(token1, token2)] = token3

    def encode(self, text):
        data = text.encode("utf-8")

        for m, token3 in self.merges.items():
            token1, token2 = m

            data = merge(data, token1, token2, token3)

        return data

    def decode(self, tokens):
        text = b"".join([self.vocab[token] for token in tokens])

        return text.decode("utf-8", errors="replace")
            
