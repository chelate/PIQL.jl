using Revise
using PIQL


gw = make_gridworld([10,10]; density = 0.1);
ctrl = make_ctrl(gw);

actor = init_tabular_actor_piql(ctrl, 1.0)
piql = initial_piql(ctrl, actor)


training_epoch!(piql, ctrl, actor)

