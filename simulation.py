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
    agent_alive: Array
    agent_vision: Array
    agent_max_energy: Array
    food_pos: Array
    key: Array
    step: int = 0


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
    agent_alive = jnp.ones(c.n_agents, dtype=bool)
    agent_vision = random.randint(key, shape=(c.n_agents,), minval=1, maxval=5)
    agent_max_energy = jnp.ones(c.n_agents) * 100

    return State(
        agent_pos=agent_pos,
        agent_energy=agent_energy,
        agent_alive=agent_alive,
        agent_vision=agent_vision,
        agent_max_energy=agent_max_energy,
        food_pos=food_pos,
        key=key,
    )


def step(s: State, c: Config) -> State:
    key, k1, k2, k3 = random.split(s.key, 4)
    offset = random.randint(k1, shape=(c.n_agents, 2), minval=-1, maxval=2)

    new_agent_pos = (s.agent_pos + offset) % jnp.array([c.columns, c.rows])
    new_agent_energy = (
        s.agent_energy - 2 - (c.vision_cost * (s.agent_vision ^ 2))
    )

    dead = s.agent_energy <= 0
    parent_idx = random.randint(k1, (c.n_agents,), 0, c.n_agents)

    new_vision = s.agent_vision[parent_idx] + random.randint(k3, (c.n_agents,), -1, 2)
    new_vision = jnp.where(new_vision > 1, new_vision, 1)

    new_pos = random.randint(k1, (c.n_agents, 2), 0, jnp.array([c.columns, c.rows]))
    agent_pos = jnp.where(dead[:, None], new_pos, new_agent_pos)
    agent_vision = jnp.where(dead, new_vision, s.agent_vision)
    agent_energy = jnp.where(dead, s.agent_max_energy, new_agent_energy)
    agent_alive = s.agent_energy > 0

    return State(
        agent_pos=agent_pos,
        agent_energy=agent_energy,
        agent_alive=s.agent_alive,
        agent_vision=agent_vision,
        agent_max_energy=s.agent_max_energy,
        food_pos=s.food_pos,
        key=key,
        step=s.step + 1,
    )
