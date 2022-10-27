# label-only-unsupervised-document-classification
phrase BERT, python-rake.

# Introduction
developed an unsupervised model that can classify texts using labels or keywords, it
mines the important phrases from the documents and embeds them using sentence BERT.
Dimensionality reduction was used to reduce the embedded vectors, where a fully connected graph was
constructed over every important phrase. The edge is defined as the euclidean distance between
low-dimension phrase embeddings We measure the minimum graph path for each phrase from single
documents to reach all keywords and select the category with minimum path sum. Results of every single
phrase in the document vote for the document class/label.
