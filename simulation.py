from jax import random, Array
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Config:
    n_agents: int = 10
    n_food: int = 5
    vision_cost: float = 0.5
    seed: int = 42
    rows: int = 30
    columns: int = 40


@dataclass(frozen=True)
class State:
    agent_pos: Array
    agent_energy: Array
    agent_vision: Array
    agent_max_energy: Array
    food_pos: Array
    key: Array
    step: int = 0

    @property
    def agent_alive(self) -> Array:
        return self.agent_energy > 0


def init_state(c: Config) -> State:
    key = random.key(c.seed)
    key, k1, k2 = random.split(key, 3)
    agent_pos = random.randint(
        k1, shape=(c.n_agents, 2), minval=0, maxval=jnp.array([c.columns, c.rows])
    )
    food_pos = random.randint(
        k2, shape=(c.n_food, 2), minval=0, maxval=jnp.array([c.columns, c.rows])
    )

    agent_energy = jnp.ones(c.n_agents) * 100
    agent_vision = random.randint(key, shape=(c.n_agents,), minval=1, maxval=5)
    agent_max_energy = jnp.ones(c.n_agents) * 100

    return State(
        agent_pos=agent_pos,
        agent_energy=agent_energy,
        agent_vision=agent_vision,
        agent_max_energy=agent_max_energy,
        food_pos=food_pos,
        key=key,
    )


def respawn(state: State, cfg: Config) -> State:
    key, k1, k2 = random.split(state.key, 3)

    dead = state.agent_energy <= 0
    parent_idx = random.randint(k1, (cfg.n_agents,), 0, cfg.n_agents)

    new_vision = state.agent_vision[parent_idx] + random.randint(
        k2, (cfg.n_agents,), -1, 2
    )
    new_vision = jnp.where(new_vision > 1, new_vision, 1)

    new_pos = random.randint(
        k1, (cfg.n_agents, 2), 0, jnp.array([cfg.columns, cfg.rows])
    )
    agent_pos = jnp.where(dead[:, None], new_pos, state.agent_pos)
    agent_vision = jnp.where(dead, new_vision, state.agent_vision)
    agent_energy = jnp.where(dead, state.agent_max_energy, state.agent_energy)

    return State(
        agent_pos=agent_pos,
        agent_energy=agent_energy,
        agent_vision=agent_vision,
        agent_max_energy=state.agent_max_energy,
        food_pos=state.food_pos,
        key=key,
        step=state.step,
    )


def step(state: State, cfg: Config) -> State:
    key, k1, k2, k3 = random.split(state.key, 4)

    if state.agent_alive.sum() / cfg.n_agents < 0.5:
        state = respawn(state, cfg)

    offset = random.randint(k1, shape=(cfg.n_agents, 2), minval=-1, maxval=2)
    new_agent_pos = (state.agent_pos + offset) % jnp.array([cfg.columns, cfg.rows])
    new_agent_energy = (
        state.agent_energy - 2 - (cfg.vision_cost * (state.agent_vision ^ 2))
    )

    return State(
        agent_pos=new_agent_pos,
        agent_energy=new_agent_energy,
        agent_vision=state.agent_vision,
        agent_max_energy=state.agent_max_energy,
        food_pos=state.food_pos,
        key=key,
        step=state.step + 1,
    )
