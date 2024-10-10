import string

# A simple char tokenizer
class Tokenizer: 
    
    def __init__(self, data):
        unique_chars = set(data)
        chars = sorted(list(unique_chars))        
        self.vocab_size = len(unique_chars)
        self.encoding_map = {c:i for i, c in enumerate(chars)}
        self.decoding_map = {i:c for i, c in enumerate(chars)}
        
    def encode(self, text): 
        return [self.encoding_map[c] for c in text]
        
    def decode(self, idx):
        return ''.join([self.decoding_map[i] for i in idx])


if __name__ == '__main__': 
    tokenizer = Tokenizer(string.ascii_lowercase)
    assert tokenizer.decode(tokenizer.encode('hello')) == 'hello'