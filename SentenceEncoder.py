import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Import the Universal Sentence Encoder's TF Hub module
# "https://tfhub.dev/google/universal-sentence-encoder/4"
# "https://tfhub.dev/google/universal-sentence-encoder-large/5"
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.KerasLayer(module_url)

# Compute a representation for each message, showing various lengths supported.
sentence1 = "I'm stuck trying to understand the code and instuctions for asynchronous loading"
sentence2 = "Want to put your code up somewhere, I'll take a look at it. Maybe (probably) can't help much, but sometimes having another set of eyes helps."
sentence3 = "-27 C this morning with windchill. I nearly died on the way to work, lol"
messages = [sentence1, sentence2, sentence3]

message_embeddings = embed.call(messages)
message_embeddings = np.array(message_embeddings).tolist()

emb1 = message_embeddings[0][:]

for i in range(0, 3, 1):
    emb2 = message_embeddings[i][:]
    cos_sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
    print("Cosine Similarity is {} ".format(cos_sim))

"""
print("Starting Loop")
for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
"""