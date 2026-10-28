# mini-gpt

Mini-GPT is a from scratch (kind of) transformer that I implemented. What that means is that I used pytorch autograd, backpropagation, abstract base classes and functions, but I implemented most of the logic like multihead attention, MLP, (sinosoidal) positional encoding, etc. on my own.

The project also contains a training loop, a jupyter notebook with testing and an equivalent gpt implementation using pytorch builtin classes that was used for testing my modules.

I had 2 main goals for this project when I started:
1. To get a deep understanding of transformers
2. To improve my pytorch skills

And along the way a 3rd goal emerged of gaining intuition for choosing hyperparameters and model choices by playing around with different versions of these models, like:
1. How big a difference does dropout make
2. What happens when I have dropout + weight decay
3. Playing around with learning rates
4. ....

Future Plans:
1. Well, first of all just finish with what I promised above :)
2. Implement my own optomizer
3. Implement Linear and Embedding layers on my own
4. Do more testing and add testing results to the README