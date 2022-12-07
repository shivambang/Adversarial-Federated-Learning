# Preventing Adversarial Attacks on Federated Learning Systems

Course Project for CNT5410 Computer and Network Security at UF

## Run

Make sure to install dependencies

> `pip install flwr torch torchvision scikit-learn`

To run a simulation

`python3 sim.py -N <num of clients> -A <num of adverseries> -D <defense>`

`defense can be one of [none, median, euclid, cosine]`

## Output

The simulator outputs the accuracy of the global model similar to-

> INFO flower 2022-12-07 16:22:49,228 | app.py:195 | app_fit: metrics_centralized {'accuracy': [(0, 0.1006), (1, 0.8119), (2, 0.9346), (3, 0.958)]}

which shows accuracies after 3 rounds of updates

## Code

The code in this repo has been extended from the [flwr framework](https://github.com/adap/flower).
