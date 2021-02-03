# %%
# 本文主要说明马尔科夫链的模型

import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# %%

seen = np.array([[0,1,0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print("The ball picked:", ", ".join(map(lambda x: observations[x[0]], seen)))
print("The hidden box", ", ".join(map(lambda x: states[x], box)))

# %%
box2 = model.predict(seen)
print("The ball picked:", ", ".join(list(map(lambda x: observations[x[0]], seen))))
print("The hidden box", ", ".join(list(map(lambda x: states[x], box2))))

# %%
# 观测序列的概率
print(model.score(seen))