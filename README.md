### Beam-GD

Many classic algorithms in statistics and machine learning rely on greedy principles: forward stepwise regression, hierarchical clustering, CART. At each step, they make the locally optimal choice, often because it's fast and works well enough. But greed has limits. Early choices constrain later options. The path you pick shapes what you can see—and what you miss.

In combinatorial problems, researchers have long explored ways to relax greediness. Beam search is one such strategy. Rather than committing to the single best option at each step, beam search keeps the top-k candidates and continues with all of them in parallel, selecting the best downstream performers. This allows for exploration without a full brute-force search.

What if we brought the same idea to gradient descent?

#### Beam Search for SGD

We implement a simple dynamic beam search variant for training a neural network classifier:

* Maintain `k` models at each step (the beam).
* Each model performs one step of SGD.
* We optionally jitter parameters (add noise) to explore nearby regions.
* Evaluate each model on a validation set.
* Keep the top-k performers.
* Repeat.

This is like running multiple SGD processes, but actively selecting the best `k` at each step based on validation performance—not just letting them run independently.

#### Does It Work?

We compared standard SGD and dynamic beam search on a synthetic classification task. Both used the same architecture and training budget. Here's the comparison:

* **Validation loss**: Beam search consistently achieves lower validation loss across epochs.
* **Test loss**: Final test loss is also lower for beam search.

```python
aseline Test Loss: 0.7224
Beam Search Test Loss: 0.6632
```

#### Why It Works

Vanilla SGD is greedy—it descends the steepest slope it sees. But that slope may lead to a local minimum, saddle point, or flat region. By maintaining multiple optimization paths and selecting based on validation loss, beam search allows:

* **Exploration**: Injecting noise expands the search.
* **Selection**: Validation loss acts as an external guide.
* **Adaptation**: Poor performers are dropped, good ones retained.

It’s a simple idea—keep more options open, then let performance decide.

#### What's Next

This beam-based approach adds very little overhead for small models, and could be extended further:

* Beam width decay over time
* Crossover or ensembling between beams
* Application to RL, transformers, or large-scale fine-tuning

Greedy is fast. But when you need better solutions, sometimes it pays to be a little less greedy.

---

Code available on request or via Colab-ready snippet.
