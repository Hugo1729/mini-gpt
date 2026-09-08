import regex as re


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


def get_cnts(chunked_data):
    ans = {}

    for chunk in chunked_data:
        for x, y in zip(chunk[:-1], chunk[1:]):
            ans[(x,y)] = ans.get((x,y), 0) + 1

    return ans

def merge(chunked_data, token1, token2, new_token):
    ans = []

    for chunk in chunked_data:
        new_chunk = []
        i = 0

        while i < len(chunk):
            if (i+1 < len(chunk) and chunk[i] == token1 and chunk[i+1] == token2):
                new_chunk.append(new_token)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1

        ans.append(new_chunk)

    return ans

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, custom_regex=None):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        self.regex = GPT4_SPLIT_PATTERN
        if (custom_regex is not None):
            self.regex = custom_regex

    def train(self, text, num_tokens):
        chunked_data = [list(chunk.encode("utf-8")) for chunk in re.findall(self.regex, text)]

        while (len(self.vocab) < num_tokens):
            cnts = get_cnts(chunked_data)

            if (not cnts):
                break

            token1, token2 = max(cnts, key = cnts.get)

            new_token = len(self.vocab)

            chunked_data = merge(chunked_data, token1, token2, new_token)

            self.vocab[new_token] = self.vocab[token1] + self.vocab[token2]
            self.merges[(token1, token2)] = new_token

    def encode(self, text):
        chunked_data = [list(chunk.encode("utf-8")) for chunk in re.findall(self.regex, text)]

        # as of python 3.7 this works well and guarantees iteration order,
        # is the same as the order we as the order we added the items in
        for pair, new_token in self.merges.items():
            token1, token2 = pair

            chunked_data = merge(chunked_data, token1, token2, new_token)

        return [token for chunk in chunked_data for token in chunk]

    def decode(self, tokens):
        raw_bytes = b"".join([self.vocab[token] for token in tokens])

        return raw_bytes.decode("utf-8", errors="replace")


bpe = Tokenizer()

bpe.train(text, 300)

print()
print("######VOCAB######")
print()

for k, v in bpe.vocab.items():
    print(k, "\"" + v.decode("utf-8", errors="replace") + "\"")

print()
print("######MERGES######")
print()

for k, v in bpe.merges.items():
    print(k, v)

print(bpe.decode(bpe.encode(text)))

print()


print(text == bpe.decode(bpe.encode(text)))
            
