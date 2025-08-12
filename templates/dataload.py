from sklearn.datasets import fetch_20newsgroups

# Load the full dataset (you can also load specific categories)
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Access the data and target
texts = newsgroups_data.data
labels = newsgroups_data.target

# Print sample
print(f"Number of articles: {len(texts)}")
print(f"Sample article:\n{texts[0]}")
