import pyray as rl
from simulation import Config, State

CELL_SIZE = 20


def iter_agents(state: State):
    for i in range(len(state.agents.pos)):
        if state.agents.alive[i]:
            yield (
                int(state.agents.pos[i, 0]),
                int(state.agents.pos[i, 1]),
                int(state.agents.vision[i]),
                int(state.agents.energy[i]),
                int(state.agents.max_energy[i]),
            )


def iter_consumables(state: State):
    for i in range(len(state.consumables.pos)):
        yield (
            int(state.consumables.pos[i, 0]),
            int(state.consumables.pos[i, 1]),
            int(state.consumables.energy[i]),
        )


def draw_agents(state: State):
    for x, y, vision, energy, max_energy in iter_agents(state):
        energy = max(0, energy / max_energy)
        color = rl.Color(255 - int(energy * 255), int(energy * 255), 0, 255)
        rl.draw_rectangle(
            x * CELL_SIZE - vision * CELL_SIZE,
            y * CELL_SIZE - vision * CELL_SIZE,
            (vision * 2 + 1) * CELL_SIZE,
            (vision * 2 + 1) * CELL_SIZE,
            rl.color_alpha(color, 0.1),
        )
        rl.draw_rectangle(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, color)


def draw_food(state: State):
    for x, y, energy in iter_consumables(state):
        rl.draw_rectangle(
            x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, rl.DARKGREEN
        )
        rl.draw_rectangle_lines(
            x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, rl.BLACK
        )


def draw_grid(cfg: Config):
    for i in range(cfg.rows):
        for j in range(cfg.columns):
            rl.draw_rectangle_lines(
                j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE, rl.GRAY
            )


def draw(state: State, cfg: Config):
    rl.clear_background(rl.DARKGRAY)

    draw_grid(cfg)
    draw_agents(state)
    draw_food(state)

    rl.draw_text(f"step: {state.step}", 10, 10, 24, rl.WHITE)
