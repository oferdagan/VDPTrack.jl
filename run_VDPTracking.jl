using Pkg
Pkg.activate(".")



using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using Random
using Plots


using VDPTracking


pomdp = VDPTrackPOMDP()


# s0 = rand(initialstate(pomdp))

# a = action(planner, s0)

# sp, o, r = @gen(:sp, :o, :r)(pomdp, s0, a)
Random.seed!(1000)

solver = POMCPOWSolver(criterion=MaxUCB(20.0))
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)

hist = simulate(hr, pomdp, planner)

planner = RandomPolicy(pomdp)
rhist = simulate(hr, pomdp, planner)
# for (s, b, a, r, sp, o) in rhist
#     # @show s, a, r, sp
#     # @show a
# end
for h in hist
    @show h.a
end


println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(discounted_reward(rhist))
        POMCPOW: $(discounted_reward(hist))
        
    """)
    # POMCPOW: $(discounted_reward(hist))

sth = []


target2plot = 3
for h in hist
    push!(sth,h.s.target[target2plot])
    # push!(sah,row[:sp].base_state.agent)
    # push!(bh, row[:b].particles)
    # push!(bph, row[:bp].particles)
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

    xt = reshape(vcat(sth...),2 , 50)
    scatter!(p1, xt[1,:], xt[2,:])
    sth[1]
@show o.obs
@show o
@show r