import random
import matplotlib.pyplot as plt

print("STEP 1: Implementations")

# 1. Brute Force Algorithm
def brute_force(text, pattern):
    n, m = len(text), len(pattern)
    positions = []
    comparisons = 0
    
    if m == 0 or n < m:
        return positions, comparisons

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            comparisons += 1
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
            
    return positions, comparisons

# 2. Knuth-Morris-Pratt (KMP) Algorithm
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    positions = []
    comparisons = 0
    
    if m == 0 or n < m:
        return positions, comparisons

    lps = compute_lps(pattern)
    i = 0 
    j = 0 
    
    while i < n:
        comparisons += 1
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == m:
                positions.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
                
    return positions, comparisons

# 3. Boyer-Moore-Horspool (BMH) Algorithm
def bmh_search(text, pattern):
    n, m = len(text), len(pattern)
    positions = []
    comparisons = 0
    
    if m == 0 or n < m:
        return positions, comparisons

    # Build the Bad Character table for the shift heuristic
    bad_char = {}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i

    i = m - 1 # text index aligned with the end of the pattern
    while i < n:
        k = 0
        while k < m:
            comparisons += 1
            # BMH compares from right to left
            if text[i - k] == pattern[m - 1 - k]:
                k += 1
            else:
                break
                
        if k == m:
            positions.append(i - m + 1)
            
        # Shift based on the character in the text that aligns with the END of the pattern
        # If it's not in the bad_char table, shift by full pattern length (m)
        shift = bad_char.get(text[i], m)
        i += shift
        
    return positions, comparisons



print("STEP 2 & 3: Testing on Different Data & Table")

text_short = "the quick brown fox jumps over the lazy dog. this is a short text used for testing algorithms."
text_long = text_short * 15 
text_dna = "".join(random.choices(["A", "C", "G", "T"], k=1000)) # 1000 length DNA string

# Defining 3 patterns for each dataset (Short pattern, long pattern, non-existent/repeating pattern)
test_cases = [
    {"name": "Short Text", "text": text_short, "patterns": ["fox", "algorithms", "zebra"]},
    {"name": "Long Text", "text": text_long, "patterns": ["lazy", "the quick brown", "missingword"]},
    {"name": "DNA Text", "text": text_dna, "patterns": ["AGCT", "AAAAA", "TGCATGCA"]}
]

# Results storage for the graph later
results_for_graph = []

# Header for our formatted table
print(f"{'Text Type':<15} | {'Pattern':<18} | {'Len(T)'} | {'Len(P)'} | {'BF Comp':<10} | {'KMP Comp':<10} | {'BMH Comp':<10}")
print("-" * 105)

for case in test_cases:
    for pattern in case["patterns"]:
        pos_bf, comp_bf = brute_force(case["text"], pattern)
        pos_kmp, comp_kmp = kmp_search(case["text"], pattern)
        pos_bmh, comp_bmh = bmh_search(case["text"], pattern)
        
        # Verify correctness (all should find the same number of matches)
        assert len(pos_bf) == len(pos_kmp) == len(pos_bmh)
        
        print(f"{case['name']:<15} | {pattern:<18} | {len(case['text']):<6} | {len(pattern):<6} | {comp_bf:<10} | {comp_kmp:<10} | {comp_bmh:<10}")
        
        results_for_graph.append({
            "label": f"{case['name'][:5]}: '{pattern}'",
            "bf": comp_bf,
            "kmp": comp_kmp,
            "bmh": comp_bmh
        })

print("STEP 4: Visualization (Opening Graph window...)")

# Extract data for plotting
labels = [r["label"] for r in results_for_graph]
bf_comps = [r["bf"] for r in results_for_graph]
kmp_comps = [r["kmp"] for r in results_for_graph]
bmh_comps = [r["bmh"] for r in results_for_graph]

x = range(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar([i - width for i in x], bf_comps, width, label='Brute Force')
bars2 = ax.bar(x, kmp_comps, width, label='KMP')
bars3 = ax.bar([i + width for i in x], bmh_comps, width, label='BMH')

ax.set_ylabel('Number of Character Comparisons')
ax.set_title('Pattern Matching Algorithm Efficiency (By Comparisons)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()                        