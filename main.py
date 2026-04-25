import render as ui
import pyray as rl
import simulation as sim

SCREEN_W = 800
SCREEN_H = 600

rl.init_window(SCREEN_W, SCREEN_H, "naco")
rl.set_target_fps(5)

config = sim.Config()
state = sim.init_state(config)

while not rl.window_should_close():
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)

    state = sim.step(state, config)
    ui.draw(state, config)

    rl.end_drawing()

rl.close_window()
