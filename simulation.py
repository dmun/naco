import jax
from jax import random, Array
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Config:
    n_agents: int = 10
    n_consumables: int = 30
    vision_cost: float = 0.1
    metabolism: float = 0.1
    seed: int = 42
    rows: int = 40
    columns: int = 40


@dataclass
class Agents:
    pos: Array
    energy: Array
    vision: Array
    max_energy: Array

    @property
    def alive(self) -> Array:
        return self.energy > 0

    @property
    def mean_vision(self) -> float:
        return float(jnp.mean(self.vision))


@dataclass
class Consumables:
    pos: Array
    energy: Array


@dataclass
class State:
    agents: Agents
    consumables: Consumables
    key: Array
    step: int = 0


def init_state(cfg: Config) -> State:
    key = random.key(cfg.seed)
    key, k1, k2 = random.split(key, 3)

    agents = Agents(
        pos=random.randint(
            k1,
            shape=(cfg.n_agents, 2),
            minval=0,
            maxval=jnp.array([cfg.columns, cfg.rows]),
        ),
        energy=jnp.ones(cfg.n_agents) * 100,
        vision=random.randint(key, shape=(cfg.n_agents,), minval=1, maxval=10),
        max_energy=jnp.ones(cfg.n_agents) * 100,
    )

    consumables = Consumables(
        pos=random.randint(
            k2,
            shape=(cfg.n_consumables, 2),
            minval=0,
            maxval=jnp.array([cfg.columns, cfg.rows]),
        ),
        energy=jnp.ones(cfg.n_consumables),
    )

    return State(agents=agents, consumables=consumables, key=key)


def respawn(agents: Agents, cfg: Config, key) -> Agents:
    key, k1, k2 = random.split(key, 3)

    parent_idx = random.randint(k1, (cfg.n_agents,), 0, cfg.n_agents)

    # temporary way of inheriting traits
    new_vision = agents.vision[parent_idx] + random.randint(k2, (cfg.n_agents,), -1, 2)
    new_vision = jnp.where(new_vision > 1, new_vision, 1)

    new_pos = random.randint(
        k1, (cfg.n_agents, 2), 0, jnp.array([cfg.columns, cfg.rows])
    )
    new_pos = jnp.where(~agents.alive[:, None], new_pos, agents.pos)

    new_vision = jnp.where(~agents.alive, new_vision, agents.vision)
    new_energy = jnp.where(~agents.alive, agents.max_energy, agents.energy)

    return Agents(
        pos=new_pos, vision=new_vision, energy=new_energy, max_energy=agents.max_energy
    )


def find_nearest_food(agent_pos, food_pos, vision_range):
    deltas = food_pos - agent_pos
    dists = jnp.abs(deltas).sum(axis=1)

    nearest_idx = jnp.argmin(dists)
    nearest_dist = dists[nearest_idx]
    nearest_dir = jnp.sign(deltas[nearest_idx])

    in_vision = nearest_dist <= vision_range
    return nearest_dir, in_vision


def agent_move(state: State, cfg: Config, key) -> State:
    nearest_dirs, in_vision = jax.vmap(find_nearest_food, in_axes=(0, None, 0))(
        state.agents.pos, state.consumables.pos, state.agents.vision
    )
    random_offset = random.randint(key, (cfg.n_agents, 2), -1, 2)
    offset = jnp.where(in_vision[:, None], nearest_dirs, random_offset)
    state.agents.pos = (state.agents.pos + offset) % jnp.array([cfg.columns, cfg.rows])
    return state


def consume(state: State, cfg: Config, key) -> State:
    agents = state.agents
    consumables = state.consumables

    overlap = (agents.pos[:, None, :] == consumables.pos[None, :, :]).all(axis=2)

    agents_on_food = overlap.any(axis=1)
    food_consumed = overlap.any(axis=0)

    new_energy = agents.energy + jnp.where(agents_on_food, 40, 0)
    random_pos = random.randint(
        key, consumables.pos.shape, 0, jnp.array([cfg.columns, cfg.rows])
    )

    state.agents.energy = jnp.minimum(new_energy, agents.max_energy)
    state.consumables.pos = jnp.where(food_consumed[:, None], random_pos, consumables.pos)
    return state


def step(state: State, cfg: Config) -> State:
    key, k1, k2, k3 = random.split(state.key, 4)

    if state.agents.alive.sum() / cfg.n_agents < 0.7:
        # if state.agents.alive.sum() == 1:
        state.agents = respawn(state.agents, cfg, k1)

    state = agent_move(state, cfg, k2)
    state = consume(state, cfg, k3)

    energy_cost = cfg.metabolism + (cfg.vision_cost * state.agents.vision**2)
    state.agents.energy = state.agents.energy - energy_cost

    state.key = key
    state.step = state.step + 1
    return state
