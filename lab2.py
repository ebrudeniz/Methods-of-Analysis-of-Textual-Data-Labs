import math
import random
from collections import Counter
import nltk

nltk.download('brown', quiet=True)

print("STEP 1: Basic introduction to n-grams")

raw_words = nltk.corpus.brown.words()
cleaned_tokens = [w.lower() for w in raw_words if w.isalnum()]

split_idx = int(len(cleaned_tokens) * 0.95)
train_tokens = cleaned_tokens[:split_idx]
test_tokens = cleaned_tokens[split_idx:]

# Vocabulary is based ONLY on the training set
vocab = set(train_tokens)
V = len(vocab)
N_train = len(train_tokens)

print(f"Training on {N_train} words. Testing on {len(test_tokens)} words. Vocabulary size: {V}")

unigram_counts = Counter(train_tokens)
bigram_counts = Counter(zip(train_tokens[:-1], train_tokens[1:]))
trigram_counts = Counter(zip(train_tokens[:-2], train_tokens[1:-1], train_tokens[2:]))

print("\nMost frequent Unigrams:", unigram_counts.most_common(5))
print("Most frequent Bigrams:", bigram_counts.most_common(5))
print("Most frequent Trigrams:", trigram_counts.most_common(5))


print("STEP 2: N-gram probability & Laplace Smoothing")

def get_unigram_prob(word):
    return (unigram_counts[word] + 1) / (N_train + V)

def get_bigram_prob(w1, w2):
    count_context = unigram_counts[w1]
    count_ngram = bigram_counts[(w1, w2)]
    return (count_ngram + 1) / (count_context + V)

def get_trigram_prob(w1, w2, w3):
    count_context = bigram_counts[(w1, w2)]
    count_ngram = trigram_counts[(w1, w2, w3)]
    return (count_ngram + 1) / (count_context + V)

print("Probability of 'york' after 'new':", get_bigram_prob('new', 'york'))
print("Probability of 'apple' after 'new':", get_bigram_prob('new', 'apple'))


print("STEP 3: Word prediction using a bigram model (Autocomplete)")

# greedy
def autocomplete_bigram(input_word):
    input_word = input_word.lower()
    best_next_word = None
    max_prob = -1.0
    
    candidates = [b[1] for b in bigram_counts.keys() if b[0] == input_word]
    
    if not candidates:
        return unigram_counts.most_common(1)[0][0]
        
    for candidate in set(candidates):
        prob = get_bigram_prob(input_word, candidate)
        if prob > max_prob:
            max_prob = prob
            best_next_word = candidate
            
    return best_next_word

print("Autocomplete for 'united':", autocomplete_bigram('united'))
print("Autocomplete for 'artificial':", autocomplete_bigram('artificial'))
print("Autocomplete for 'good':", autocomplete_bigram('good'))


print("STEP 4: Creating a text generator (Trigram)")

# sampling but algorithm behaves like greedy because of data sparsity
def generate_text_sampling(seed_word, num_words=15):
    seed_word = seed_word.lower()
    sentence = [seed_word]
    
    # Generate 2nd word using Bigram sampling
    candidates_bg = [b[1] for b in bigram_counts.keys() if b[0] == seed_word]
    if not candidates_bg:
        second_word = unigram_counts.most_common(1)[0][0]
    else:
        probs = [get_bigram_prob(seed_word, c) for c in candidates_bg]
        # random.choices picks a word based on its probability distribution
        second_word = random.choices(candidates_bg, weights=probs, k=1)[0]
    
    sentence.append(second_word)
    
    # Generate the rest using Trigram sampling
    for _ in range(num_words - 2):
        w1, w2 = sentence[-2], sentence[-1]
        
        candidates_tg = [t[2] for t in trigram_counts.keys() if t[0] == w1 and t[1] == w2]
        
        if not candidates_tg:
            # Fallback to Bigram sampling
            candidates_bg2 = [b[1] for b in bigram_counts.keys() if b[0] == w2]
            if not candidates_bg2:
                next_word = unigram_counts.most_common(1)[0][0]
            else:
                probs2 = [get_bigram_prob(w2, c) for c in candidates_bg2]
                next_word = random.choices(candidates_bg2, weights=probs2, k=1)[0]
        else:
            probs3 = [get_trigram_prob(w1, w2, c) for c in candidates_tg]
            next_word = random.choices(candidates_tg, weights=probs3, k=1)[0]
            
        sentence.append(next_word)
        
    return " ".join(sentence)

print(f"Generated Text from 'she': {generate_text_sampling('she')}")
print(f"Generated Text from 'computer': {generate_text_sampling('computer')}")


print("STEP 5: Model evaluation using perplexity")

def compute_perplexity(tokens, model_type="bigram"):
    N = len(tokens)
    log_prob_sum = 0.0
    
    for i in range(N):
        if model_type == "unigram":
            prob = get_unigram_prob(tokens[i])
            
        elif model_type == "bigram":
            if i == 0:
                prob = get_unigram_prob(tokens[i])
            else:
                prob = get_bigram_prob(tokens[i-1], tokens[i])
                
        elif model_type == "trigram":
            if i == 0:
                prob = get_unigram_prob(tokens[i])
            elif i == 1:
                prob = get_bigram_prob(tokens[i-1], tokens[i])
            else:
                prob = get_trigram_prob(tokens[i-2], tokens[i-1], tokens[i])
                
        log_prob_sum += math.log2(prob)
        
    avg_log_prob = log_prob_sum / N
    perplexity = math.pow(2, -avg_log_prob)
    return perplexity

small_test_set = test_tokens[:1000]

pp_unigram = compute_perplexity(small_test_set, "unigram")
pp_bigram = compute_perplexity(small_test_set, "bigram")
pp_trigram = compute_perplexity(small_test_set, "trigram")

print(f"Test Set Perplexity (Unigram): {pp_unigram:.2f}")
print(f"Test Set Perplexity (Bigram):  {pp_bigram:.2f}")
print(f"Test Set Perplexity (Trigram): {pp_trigram:.2f}")
print("\nConclusion: As N increases, perplexity generally drops, indicating better predictive performance due to richer context.")