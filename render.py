import pyray as rl
from simulation import Config, State

CELL_SIZE = 20


def draw_agents(state: State, cfg: Config):
    for i in range(cfg.n_agents):
        if state.agent_alive[i]:
            vision = int(state.agent_vision[i])
            energy = max(
                0, float(state.agent_energy[i]) / float(state.agent_max_energy[i])
            )
            x = int(state.agent_pos[i][0]) * CELL_SIZE
            y = int(state.agent_pos[i][1]) * CELL_SIZE
            color = rl.Color(255 - int(energy * 255), int(energy * 255), 0, 255)
            rl.draw_rectangle(
                x - vision * CELL_SIZE,
                y - vision * CELL_SIZE,
                (vision * 2 + 1) * CELL_SIZE,
                (vision * 2 + 1) * CELL_SIZE,
                rl.color_alpha(color, 0.1),
            )
            rl.draw_rectangle(x, y, CELL_SIZE, CELL_SIZE, color)


def draw_food(state: State, cfg: Config):
    for i in range(cfg.n_food):
        x = int(state.food_pos[i][0]) * CELL_SIZE
        y = int(state.food_pos[i][1]) * CELL_SIZE
        rl.draw_rectangle(x, y, CELL_SIZE, CELL_SIZE, rl.DARKGREEN)
        rl.draw_rectangle_lines(x, y, CELL_SIZE, CELL_SIZE, rl.BLACK)


def draw_grid(cfg: Config):
    for i in range(cfg.rows):
        for j in range(cfg.columns):
            rl.draw_rectangle_lines(
                j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE, rl.GRAY
            )


def draw(state: State, cfg: Config):
    rl.clear_background(rl.DARKGRAY)

    draw_grid(cfg)
    draw_agents(state, cfg)
    draw_food(state, cfg)

    rl.draw_text(f"step: {state.step}", 10, 10, 24, rl.WHITE)
