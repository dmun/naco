import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation import Config, init_state, step

STEPS = 3000
cfg = Config()

steps = []
mean_visions = []

state = init_state(cfg)
for i in tqdm(range(STEPS)):
    state = step(state, cfg)
    steps.append(state.step)
    mean_visions.append(state.agents.mean_vision)

fig, ax = plt.subplots()

ax.plot(steps, mean_visions)
ax.set_ylabel("mean vision")

plt.tight_layout()
plt.savefig("plot.png")
plt.show()
