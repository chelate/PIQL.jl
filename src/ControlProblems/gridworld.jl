#include("agent.jl")

struct Gridworld
    grid::Array{Int64,2}
    actions::Array{Int64,1}
    terminal_cond::Array{Bool,2}
end



function initilaize_gridwold(prams::Tuple)
    initilaize_gridwold(prams[1],prams[2],prams[3],prams[4])

end

function initilaize_gridwold(
    size,
    walls,
    reward_pos::Tuple{Int64,Int64},
    rewards::Array{Int64,1},
)
    grid = convert.(Int64, ones(Int64, size) .* rewards[1])
    terminal_cond=convert.(Bool,zeros(Int8, size))
    #initilize border
    grid[1, :] .= 0
    grid[size[1], :] .= 0
    grid[:, 1] .= 0
    grid[:, size[2]] .= 0

    #initilize walls
    for wall in walls
        pos = give_grid_posistion(grid, wall)
        grid[pos] = 0
    end

    #set reward
    grid[give_grid_posistion(grid, reward_pos)] = rewards[2]
    terminal_cond[give_grid_posistion(grid, reward_pos)] = true

    #initilize possible actions (move  [up,right, down, left])
    actions = [-1, size[1], 1, -size[1]]

    return Gridworld(grid, actions,terminal_cond)
end


function build_pickle_grid(size, holes, thick, basevalue)
    pickle = convert.(Int64, ones(Int64, size) .* basevalue)
    pickle[1, :] .= 0
    pickle[size[1], :] .= 0
    pickle[:, 1] .= 0
    pickle[:, size[2]] .= 0

    for xx = 1:size[1]
        for yy = 1:size[1]
            if (yy < xx - thick)
                pickle[xx, yy] = 0
            elseif (yy > xx + thick)
                pickle[xx, yy] = 0
            end
        end
    end

    for hole = 1:holes
        point = start_random_actor(pickle, basevalue)
        pickle[point] = 0
    end
    return pickle

end


function initilaize_gridwold_pickle(
    size,
    holes,
    thick,
    reward_pos::Tuple{Int64,Int64},
    rewards::Array{Int64,1},
)
    grid = build_pickle_grid(size, holes, thick, rewards[1])

    #set reward
    grid[give_grid_posistion(grid, reward_pos)] = rewards[2]

    #initilize possible actions (move  [up,right, down, left])
    actions = [-1, size[1], 1, -size[1]]

    return Gridworld(grid, actions)
end



#### Initilazation of some paractical environt functions

function give_grid_posistion(grid, (x, y))
    return length(grid[:, 1]) * (x - 1) + y
end

function give_cartisian_posistion(grid, pos)
    l1 = length(grid[:, 1])
    y = (pos - 1) % l1
    y += 1
    x = floor(Int, (pos - y) / l1) + 1
    return x, y
end
