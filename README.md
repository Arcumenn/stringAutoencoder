# stringAutoencoder

The Python script `trainEmbedding.py` contains an implementation of an autoencoder for the words in data/asjp19Clustered.csv. It is adapted from an tutorial on automatic translation with Deep Learning by Sean Robertson (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

The task is to port this script to Julia, using `Flux` for the neural networks and `Cuda`for GPU access.