import re
from collections import Counter
import nltk

nltk.download('brown', quiet=True)

# STEP 1: Data Preparation
raw_text = nltk.corpus.brown.raw()
clean_text = re.sub(r'\s+', ' ', raw_text.lower()).strip()
words = clean_text.split()


print("--- Step 1: Data Preparation ---")

print(f"Total words:  {len(words)}")
print(f"Unique words: {len(set(words))}")
print(f"Top 20 words: {Counter(words).most_common(20)}")

# STEP 2: Implementation of Word-level BPE
# we represent each word as a tuple of characters plus a special end-of-word token "</w>" to distinguish between different words.
vocab = Counter([tuple(word) + ("</w>",) for word in words])
word_merge_rules = []

for _ in range(500):
    pair_freqs = Counter()
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pair_freqs[(word_tuple[i], word_tuple[i+1])] += freq

    if not pair_freqs:
        break

    best_pair = pair_freqs.most_common(1)[0][0]
    word_merge_rules.append(best_pair)
    a, b = best_pair

    # Merge the best pair in the vocabulary
    new_vocab = {}
    for word_tuple, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i+1] == b:
                new_word.append(a + b)
                i += 2  
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    vocab = new_vocab


print(f"\n--- Step 2: Word-level BPE ---")
print(f"First 10 merge rules: {word_merge_rules[:10]}")

# STEP 3: Implementation of Byte-level BPE
byte_sequence = list(clean_text)
byte_merge_rules = []

for _ in range(500):
    pair_freqs = Counter(zip(byte_sequence[:-1], byte_sequence[1:]))

    if not pair_freqs:
        break

    best_pair = pair_freqs.most_common(1)[0][0]
    byte_merge_rules.append(best_pair)
    a, b = best_pair

    new_seq = []
    i = 0
    while i < len(byte_sequence):
        if i < len(byte_sequence) - 1 and byte_sequence[i] == a and byte_sequence[i+1] == b:
            new_seq.append(a + b)
            i += 2
        else:
            new_seq.append(byte_sequence[i])
            i += 1
    byte_sequence = new_seq


print(f"\n--- Step 3: Byte-level BPE ---")
print(f"First 10 merge rules: {byte_merge_rules[:10]}")

# STEP 4: Comparison of Both Approaches
def apply_word_bpe(text, rules):
    tokens = []
    for word in text.split():
        word_tuple = tuple(word) + ("</w>",)
        for a, b in rules:
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i+1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_word)
        tokens.extend(list(word_tuple))
    return tokens

def apply_byte_bpe(text, rules):
    tokens = list(text)
    for a, b in rules:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

test_text = "the modern algorithms are understanding complex unstructured data"

word_tokens = apply_word_bpe(test_text, word_merge_rules)
byte_tokens = apply_byte_bpe(test_text, byte_merge_rules)

num_chars = len(test_text)
num_words = len(test_text.split())

word_cross = sum(1 for t in word_tokens if ' ' in t)
byte_cross = sum(1 for t in byte_tokens if ' ' in t)


print(f"\n--- Step 4: Comparison ---")

print(f"Test text: '{test_text}'\n")
print(f"{'Metric':<35} {'Word BPE':>12} {'Byte BPE':>12}")
print("-" * 60)
print(f"{'Total tokens':<35} {len(word_tokens):>12} {len(byte_tokens):>12}")
print(f"{'Tokens per 1000 chars':<35} {len(word_tokens)/num_chars*1000:>12.2f} {len(byte_tokens)/num_chars*1000:>12.2f}")
print(f"{'Tokens per word':<35} {len(word_tokens)/num_words:>12.2f} {len(byte_tokens)/num_words:>12.2f}")
print(f"{'Cross-word tokens':<35} {word_cross:>12} {byte_cross:>12}")

if len(word_tokens) > 0:
    print(f"{'Cross-word ratio':<35} {word_cross/len(word_tokens)*100:>11.1f}% {byte_cross/len(byte_tokens)*100:>11.1f}%")

print("\n--- Qualitative Comparison ---")
for w in ["understanding", "algorithms", "complex", "data"]:
    print(f"\n  '{w}'")
    print(f"    Word BPE: {apply_word_bpe(w, word_merge_rules)}")
    print(f"    Byte BPE: {apply_byte_bpe(' ' + w, byte_merge_rules)}")