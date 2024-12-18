simple math. takes the three power laws in the openai scaling paper (https://arxiv.org/pdf/2001.08361) and lays them out so you can quickly ballpark your own setup and see how 
much each one of the main contributing factors of model performance (compute, dataset size, model size) is holding your loss back. can also be used to determine whether or not 
you suck at writing transformer architecture.

For instance, if you have 0.125 pflop days compute, 1 million tokens of data, and a 14 billion non_embedding parameter model, this program will give you an estimated loss of
5.4272 and tell you that your dataset is tiny. this does mean you need to increase your dataset size, but it also means that if you write a model with 14m non-embedding
parameters and train it on 1 million tokens with 0.125 pfhrs compute and get a loss value higher than 5.4272, you could improve your model architecture somehow.

<img width="901" alt="Screenshot 2024-12-03 at 9 27 46 PM" src="https://github.com/user-attachments/assets/423ead16-2734-4e7c-9a40-2f1b25acd6e7">
