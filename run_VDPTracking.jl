using Pkg
Pkg.activate(".")



using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using Random
using Plots
using ParticleFilters
using LinearAlgebra
using MCTS
using Distributions

using VDPTracking

mutable struct rolloutPolicy{P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    p::P
    updater::U
end
rolloutPolicy(p::Union{POMDP,MDP}; updater=NothingUpdater()) = rolloutPolicy(p, updater)


function POMDPs.action(policy::rolloutPolicy, s)
    return TrackAction(argmax(policy.p.mdp.mu))

end


function POMDPs.updater(policy::rolloutPolicy)
    return policy.updater
end



pomdp = VDPTrackPOMDP()

n_p=1000
# s0 = rand(initialstate(pomdp))

# a = action(planner, s0)

# sp, o, r = @gen(:sp, :o, :r)(pomdp, s0, a)
Random.seed!(1000)

# solver = POMCPOWSolver(criterion=MaxUCB(100.0), tree_queries=1_000, estimate_value = RolloutEstimator(rolloutPolicy(pomdp)))
solver = POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=1_000, k_action = 28, k_observation = 28, max_depth = 30, estimate_value = RolloutEstimator(rolloutPolicy(pomdp)))


planner = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100)
up = BootstrapFilter(pomdp, n_p)
# hist = simulate(hr, pomdp, planner, up)

randomPlanner = RandomPolicy(pomdp)
# rhist = simulate(hr, pomdp, planner)
# for (s, b, a, r, sp, o) in rhist
#     # @show s, a, r, sp
#     # @show a
# # end
# for (h,h2) in zip(hist,rhist)
#     @show h.a
#     @show h2.a
# end
r_random = []
r_pomcpow = []
Nsim = 100

for n=1:Nsim
    hist = simulate(hr, pomdp, planner, up)
    rhist = simulate(hr, pomdp, randomPlanner)
    push!(r_random, discounted_reward(rhist))
    push!(r_pomcpow, discounted_reward(hist))



end

println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(mean(r_random)), $(std(r_random)/Nsim)

        POMCPOW: $(mean(r_pomcpow)), $(std(r_pomcpow)/Nsim)
        
    """)
    # POMCPOW: $(discounted_reward(hist))


colorList = [:red, :blue, :green, :orange, :cyan]

target2plot = 1
sth = []
bph = []

for h in hist
    push!(sth,h.s.target[target2plot])
  
    # println(h.s.target)
    # println(h.s.target[target2plot])
    # push!(sah,row[:sp].base_state.agent)
    # push!(bh, row[:b].particles)
    push!(bph, h.b.particles)
end



p1 = plot(
     
    legend=:bottomleft, 
    legendfontsize=18, 
    grid = true,
    # aspect_ratio=:equal,
    size=(500,500),
    dpi = 100,
    # legend=true,
    gridlinewidth=2.0,
    # gridstyle=:dash,
    axis=true,
    gridalpha=0.0,
    # xticks=x_axis,
    tickfontsize=8,
    xticks=collect(-4.0:1.0:4.0),
    yticks=collect(-4.0:1.0:4.0),
    xlabel="X",
    ylabel="Y",
    title=" ",
    )

    xt = reshape(vcat(sth...),2 , 75)
    scatter!(p1, xt[1,:], xt[2,:], color=colorList[target2plot])
    
display(p1)

p2 = plot(
     
    legend=:bottomright, 
    legendfontsize=18, 
    grid = true,
    # aspect_ratio=:equal,
    size=(500,500),
    dpi = 100,
    # legend=true,
    gridlinewidth=2.0,
    # gridstyle=:dash,
    axis=true,
    gridalpha=0.0,
    # xticks=x_axis,
    tickfontsize=8,
    # xticks=collect(-4.0:1.0:4.0),
    yticks=collect(-4.0:1.0:4.0),
    xlabel="X",
    ylabel="Y",
    title=" ",
    )

    # xt = reshape(vcat(sth...),2 , 50)
    scatter!(p2, norm.(sth), color=colorList[target2plot])




    # anim = @animate for idx=1:1:100
    #     p = plot(
    #     # pomdp, 
    #     legend=:bottomleft, 
    #     legendfontsize=18, 
    #     grid = true,
    #     # aspect_ratio=:equal,
    #     size=(500,500),
    #     dpi = 100,
    #     # legend=true,
    #     gridlinewidth=2.0,
    #     # gridstyle=:dash,
    #     axis=true,
    #     gridalpha=0.0,
    #     # xticks=x_axis,
    #     tickfontsize=8,
    #     xticks=collect(-4.0:1.0:4.0),
    #     yticks=collect(-4.0:1.0:4.0),
    #     xlabel="X",
    #     ylabel="Y",
    #     title="k = $idx",
    #     )
         
    
    #     x_t, y_t = sth[idx]
    
    #     b_t = []
       
    #     for b in bph[idx]
    #         push!(b_t, b.target[target2plot])
            
    #     end
    #     b_t = reshape(vcat(b_t...), 2, n_p)
    #     xlims!(-4,4)
    #     ylims!(-4,4)
    
    #     # plot!(clear=true)
    #     scatter!(p,  [x_t],[y_t], label="target", color=:orange, markersize=10)
    #     scatter!(p, xt[1,:], xt[2,:], color=colorList[target2plot])
    #     scatter!(p, b_t[1,:], b_t[2, :], color=colorList[target2plot+1], markersize=4)
        
        
        
    #     # push!(p)
    
    
    #     # push!(p, 1, sin(x))
    # end
    # gif(anim, "animation.gif", fps = 2)