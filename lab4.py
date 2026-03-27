import time
from collections import Counter
import nltk

nltk.download('brown', quiet=True)

print("PART 1: Computation of Edit Distance")
print("==========================================")

def levenshtein_distance(s1, s2):
    """Computes the Levenshtein distance between two strings using Dynamic Programming."""
    m, n = len(s1), len(s2)
    
    # Create an empty (m+1) x (n+1) matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the first row and first column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Characters match, no operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Deletion
                    dp[i][j - 1],    # Insertion
                    dp[i - 1][j - 1] # Substitution
                )
    return dp[m][n]

# Test on own data
word1, word2 = "kitten", "sitting"
print(f"Levenshtein distance between '{word1}' and '{word2}' is: {levenshtein_distance(word1, word2)}")
print(f"Expected: 3 (k->s, e->i, +g)")

print()
print("PART 2: Automatic Word Correction (Norvig's Approach)")
print("==========================================")

# 1. Dictionary Preparation
# Extract words, convert to lowercase, keep only alphabetic words
words = [w.lower() for w in nltk.corpus.brown.words() if w.isalpha()]
WORDS = Counter(words)
N_WORDS = sum(WORDS.values())

def probability(word):
    """Probability of word occurrence: P(word) = Count(word) / Total_Words"""
    return WORDS[word] / N_WORDS

# 2. Generating Word Variants
def edits1(word):
    """Generate all word variants with an edit distance of 1."""
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1] # Adjacent swap
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    
    # Use a set to avoid duplicate variants!
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    """Generate all word variants with an edit distance of 2."""
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words_list):
    """Return the subset of words that actually exist in our dictionary."""
    return set(w for w in words_list if w in WORDS)

def get_candidates(word):
    """Generate possible correction candidates for a word."""
    # Priority: 1. Known word, 2. Known distance-1 edits, 3. Known distance-2 edits, 4. Original word
    candidates = known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
    return candidates

def correct_word(word):
    """Select the most probable word from generated candidates."""
    candidates = get_candidates(word)
    return max(candidates, key=probability)

# 3. Selecting the Most Probable Word (Testing the Sentence)
sentence = "I well go to the restorant tonigth and order a delicshus desert befor going home"
print(f"Original Sentence: {sentence}")

corrected_words = []
for w in sentence.split():
    corrected_words.append(correct_word(w.lower()))
    
corrected_sentence = " ".join(corrected_words)
print(f"Corrected Sentence: {corrected_sentence}")

test = "restaurant"
print(f"\nthe number of variants for '{test}':")
print(f"Edit Distance 1 variants: {len(edits1(test))}")
print(f"Edit Distance 2 variants: {len(edits2(test))}")

print()
print("PART 3: Alternative Approach & Efficiency Comparison")
print("==========================================")

def brute_force_correct(target_word):
    """Brute-force approach: Calculate edit distance for every word in the dictionary."""
    valid_candidates = []
    
    for dict_word in WORDS:
        # Optimization: Only compute if length difference is <= 2
        if abs(len(target_word) - len(dict_word)) <= 2:
            dist = levenshtein_distance(target_word, dict_word)
            if dist <= 2:
                valid_candidates.append(dict_word)
                
    if not valid_candidates:
        return target_word
        
    return max(valid_candidates, key=probability)

# Comparison Test
test_word = "delicshus"

# Time the Generator approach
start_time = time.time()
gen_result = correct_word(test_word)
gen_time = time.time() - start_time

# Time the Brute Force approach
start_time = time.time()
bf_result = brute_force_correct(test_word)
bf_time = time.time() - start_time

print(f"Testing misspelled word: '{test_word}'")
print(f"1. Generation Approach Result: '{gen_result}' | Time: {gen_time:.4f} seconds")
print(f"2. Brute-Force Approach Result: '{bf_result}' | Time: {bf_time:.4f} seconds")