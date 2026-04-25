import jax
from jax import random, Array
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Config:
    n_agents: int = 10
    n_consumables: int = 5
    vision_cost: float = 0.2
    metabolism: int = 1
    food_value: int = 20
    seed: int = 42
    rows: int = 30
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
        vision=random.randint(key, shape=(cfg.n_agents,), minval=1, maxval=5),
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


def agent_move(state: State, cfg: Config, key) -> Array:
    agents = state.agents
    offset = random.randint(key, (cfg.n_agents, 2), -1, 2)
    return (agents.pos + offset) % jnp.array([cfg.columns, cfg.rows])


def step(state: State, cfg: Config) -> State:
    agents = state.agents
    key, k1, k2 = random.split(state.key, 3)

    if agents.alive.sum() / cfg.n_agents < 0.5:
        agents = respawn(agents, cfg, k1)

    new_agent_energy = (
        agents.energy - cfg.metabolism - (cfg.vision_cost * agents.vision**2)
    )
    new_agent_pos = agent_move(state, cfg, k2)

    return State(
        agents=Agents(
            new_agent_pos,
            new_agent_energy,
            agents.vision,
            agents.max_energy,
        ),
        consumables=state.consumables,
        key=key,
        step=state.step + 1,
    )
