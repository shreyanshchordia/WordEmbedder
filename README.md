# WordEmbedder

Importing the WordEmbedder can simplify a lot of necessary preprocessing tasks that need to be done when working on NLP Projects. 

Tasks like:
  
   1. Getting embeddings of words in the dataset
   
   2. Generating vocabulary for your tokenized sentences
   
   3. Converting tokenized sentences in number sequences 
   
   4. Retrieving the tokenized sentences back from the number sequences
   
   5. Generate an embedding matrix for the vocabulary that is demanded by the Embedding Layer from tensorflow.keras.layers as the weights parameter when using pre-trained embeddings.
   
  consume a lot of our time and lines of code. 
  
  All these tasks can individually be done in a single line of code by importing the `WordEmbedder` script.
  
  Other than the above mentioned tasks there are few other features that the script provides as well:
  
   1. Getting K most similar words to a given word
   2. Getting K top words that satisfy an analogy
   3. Getting cosine similarity between two words
   4. Getting the cosine similarity score for a 4-word analogy

# How To Use The WordEmbedder

The notebook `how_to_use_WordEmbedder.ipynb` in the repository specifies all the details on how to use the `WordEmbedder`.

### Sample Code
    
    #### Initiating an object of Embedder class
    emb = Embedder(dimensions=50) # dimensions can be 50, 100, 200 or 300 # This line take a while to execute when run for the first time because the embeddings are downloaded
    
    #### This embedder maps words to their embeddings
    
    embedder = emb.get_embedder()
    print(embedder['beautiful']) 
    
    #### Generating Vocabulary
    vocab, vocab_dict = generate_vocabulary(tokenized_sentences) # To generate vocabulary for the tokenized sentences
    
    #### Generating the embedding_matrix
    embedding_matrix, id2token, token2id = emb.get_embedding_matrix(vocab, num_words=10)
    
    #### Generating number sequences from tokenized sentences
    num_sequences = token_seq_to_num_seq(tokenized_sentences, token2id, oov_token='<oov>')
    
    #### Retrieving  tokenized sentences from number sequences
    retrieved_tokenized_sequences = num_seq_to_token_seq(num_sequences, id2token)
    
   Refer `how_to_use_WordEmbedder.ipynb` for the other features.
   
   Open to contributions! :)
