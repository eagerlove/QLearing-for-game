# Introduction
![License](https://img.shields.io/badge/license-MIT-green)

 The project developed an AI assistant for the game "The Three Kingdoms Killing" that facilitates "the process of traversing a thousand miles and riding alone". It employed the Q-learning method to efficiently navigate the global map, enabling players to acquire rewards more swiftly.

# Run

```sh
prompt> pip install -r requirements.txt
prompt> python train.py
```

# Virtualization
After the training, use the following command to view the training process diagram.
```sh
prompt> tensorborad --logdir=./runs
```

# Tips
Due to the limited ability of the editor, the auxiliary AI can only solve the previous levels where only carrot items exist.
Like this:
![Preview](./example.jpg?sanitize=true)

# Have a good time!
