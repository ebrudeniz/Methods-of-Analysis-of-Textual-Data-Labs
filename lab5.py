import re
from sklearn.datasets import fetch_20newsgroups

print("\nInverted Index Construction & Stats\n")

# 1. Corpus oluşturma (Hızlı çalışması için 500 döküman)
dataset = fetch_20newsgroups(subset='train', categories=['sci.space', 'comp.graphics'])
documents = dataset.data[:500] 

STOPWORDS = {"the", "a", "an", "and", "or", "to", "is", "in", "of", "for", "with", "on", "it", "this", "that", "be", "are", "as"}

inverted_index = {}
doc_lengths = {} 
ALL_DOCS = set(range(len(documents))) # all document IDs

# indexing
for doc_id, text in enumerate(documents):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    valid_tokens = [t for t in tokens if t not in STOPWORDS]
    doc_lengths[doc_id] = len(valid_tokens)
    
    for token in valid_tokens:
        if token not in inverted_index:
            inverted_index[token] = {}
        if doc_id not in inverted_index[token]:
            inverted_index[token][doc_id] = 1
        else:
            inverted_index[token][doc_id] += 1

num_tokens = len(inverted_index)
total_entries = sum(len(p_list) for p_list in inverted_index.values())
avg_posting_length = total_entries / num_tokens if num_tokens > 0 else 0

print("Index built successfully!")
print(f"Number of unique tokens: {num_tokens}")
print(f"Total entries: {total_entries}")
print(f"Average posting list length: {avg_posting_length:.2f}")


'''Query Evaluator & Extended Model'''

def evaluate_boolean_query(query):
    tokens = re.findall(r'\(|\)|AND|OR|NOT|\b[a-zA-Z]+\b', query)
    
    eval_string = ""
    sets_dict = {'ALL_DOCS': ALL_DOCS} 
    
    for t in tokens:
        if t in ["AND", "OR", "NOT", "(", ")"]:
            if t == "AND": eval_string += " & "
            elif t == "OR": eval_string += " | "
            elif t == "NOT": eval_string += " ALL_DOCS - "
            else: eval_string += t
        else:
            t_lower = t.lower()
            set_name = f"set_{t_lower}"
            sets_dict[set_name] = set(inverted_index.get(t_lower, {}).keys())
            eval_string += set_name

    try:
        result_set = eval(eval_string, {"__builtins__": None}, sets_dict)
        return result_set
    except Exception:
        return None # return None for invalid queries

def custom_extended_boolean(query):
    matched_docs = evaluate_boolean_query(query)
    
    if not matched_docs: # no matches or invalid query
        return []
    
    # 
    raw_tokens = re.findall(r'\b[a-zA-Z]+\b', query)
    query_terms = [t.lower() for t in raw_tokens if t not in ["AND", "OR", "NOT"]]
    
    scored_docs = []
    
    for doc_id in matched_docs:
        score = 0
        for term in query_terms:
            if term in inverted_index and doc_id in inverted_index[term]:
                score += inverted_index[term][doc_id]
        
        # density normalization: score / doc_length
        final_score = score / doc_lengths[doc_id] if doc_lengths[doc_id] > 0 else 0
        scored_docs.append((doc_id, final_score))
    
    # sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs



print("\nInteractive Query Interface")
print("======================================")

print("Type 'exit' to stop the program.")
print("Note: Logical operators (AND, OR, NOT) must be UPPERCASE.")

while True:
    user_q = input("\nEnter your Boolean query: ").strip()
    
    if user_q.lower() == 'exit':
        print("Closing the search engine. Goodbye!")
        break
    if not user_q:
        continue
        
    print("\n--- Standard Boolean Search ---")
    standard_results = evaluate_boolean_query(user_q)
    
    if standard_results is None:
        print("Error: Invalid query syntax. Please check your spelling and parentheses.")
        continue
        
    print(f"Found {len(standard_results)} document(s).")
    
    if standard_results:
        matched_ids = list(standard_results)
        # Yönerge: "Display at least document IDs... from each matched document"
        print(f"Matched Document IDs: {matched_ids}")
        
        # Sadece fikir vermesi için ilk dökümanın kısa bir önizlemesini bırakıyoruz
        preview = documents[matched_ids[0]][:80].replace('\n', ' ')
        print(f"Preview of First Match (Doc {matched_ids[0]}): {preview}...")
        
        print("\n--- Extended Boolean (Ranked) Search ---")
        ranked_results = custom_extended_boolean(user_q)
        print("Top 3 most relevant documents:")
        for doc_id, score in ranked_results[:3]:
            print(f"  Doc ID: {doc_id} | Relevance Score: {score:.5f}")