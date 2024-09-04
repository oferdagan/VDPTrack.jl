module VDPTracking

# package code goes here
using POMDPs
using StaticArrays
using Parameters
using Plots
using Distributions
using POMDPTools
using ParticleFilters
using Random
using LinearAlgebra


const Vec2 = SVector{2, Float64}
const Vec8 = SVector{8, Float64}

# importall POMDPs
import Base: rand, eltype, convert
import MCTS: next_action, n_children
import ParticleFilters: obs_weight
import POMDPs: actions, isterminal

export
    TrackState,
    TrackAction,
    VDPTrackMDP,
    VDPTrackPOMDP,
    Vec2,

    
    convert_s,
    convert_a,
    convert_o,
    obs_weight,

    ToNextML,
    ToNextMLSolver,
    NextMLFirst,
    DiscretizedPolicy,
    ManageUncertainty,
    # CardinalBarriers,
    mdp,
    isterminal,
    next_ml_target

struct TrackState
    target::Vector{Vec2}
    t::Int
end

struct TrackObs
    obs::Vec8
    detect::Bool
end

const TrackAction = CartesianIndex{1}

# struct TrackAction
#     look::Int    # which object to look at
# end

@with_kw struct VDPTrackMDP <: MDP{TrackState, Float64}
    mu::Vector{Float64}     = [1.5, 2.0, 0.5]
    dt::Float64             = 0.1
    step_size::Float64      = 0.1
    pos_std::Float64        = 0.00005
    track_terminate::Bool   = true
    maxTimeSteps::Int64     = 50
    discount::Float64       = 0.95
end

@with_kw struct VDPTrackPOMDP <: POMDP{TrackState, TrackAction, TrackObs}
    mdp::VDPTrackMDP            = VDPTrackMDP()
    # meas_cost::Float64          = 5.0
    active_meas_std::Float64    = 0.05
    p_detect::Vector{Float64}   = [0.7, 0.6, 0.9] #[1, 1, 1]    # Probability of detection
    N_obj::Int64                = 3
    meas_std::Float64           = 0.05
end

const VDPTrackProblem = Union{VDPTrackMDP,VDPTrackPOMDP}
mdp(p::VDPTrackMDP) = p
mdp(p::VDPTrackPOMDP) = p.mdp

function next_ml_target(p::VDPTrackMDP, pos::Vec2, obj::Int64)
    steps = round(Int, p.step_size/p.dt)
    for i in 1:steps
        pos = rk4step(p, pos, obj)
    end
    return pos
end
next_ml_target(p::VDPTrackMDP, pos::AbstractVector, obj::Int64) = next_ml_target(p, convert(Vec2, pos), obj)

function POMDPs.transition(pp::VDPTrackProblem, s::TrackState, a::TrackAction)
    ImplicitDistribution(pp, s, a) do pp, s, a, rng
        p = mdp(pp)

        targ = []
        i= 1
        for obj in s.target
            push!(targ, next_ml_target(p, obj, i) + p.pos_std*SVector(randn(rng), randn(rng)) )
            i += 1
        end

        
        return TrackState(targ, s.t+1)
    end
end

function POMDPs.reward(pp::VDPTrackProblem, s::TrackState, a::TrackAction, sp::TrackState, o::TrackObs)
    p = mdp(pp)
    d = norm(s.target[a])


    if s.t == p.maxTimeSteps
        return 0.0
    else
        return o.detect*d/2
    end
    
end

POMDPs.discount(pp::VDPTrackProblem) = mdp(pp).discount
isterminal(pp::VDPTrackProblem, s::TrackState) = mdp(pp).track_terminate && s.t == mdp(pp).maxTimeSteps


function POMDPs.actions(::VDPTrackPOMDP) 
    return CartesianIndices(ntuple(Returns(1:3), 1))
end

function POMDPs.reward(p::VDPTrackPOMDP, s::TrackState, a::TrackAction, sp::TrackState)
    return reward(mdp(p), s, a, sp) 
end

#=
Beam | covers (deg)
-------------------
1    | (0,45]
2    | (45,90]
etc.
=#

struct BeamDist
    abeam::Int
    an_detect::Normal{Float64}
    an::Normal{Float64}
    n::Normal{Float64}
    p_detect::Bernoulli{Float64}
end

function rand(rng::AbstractRNG, d::BeamDist)
    o = MVector{8, Float64}(undef)
    detect = rand(rng, d.p_detect) 
    for i in 1:length(o)
        if i == d.abeam
            if detect
                o[i] = rand(rng, d.an_detect)
            else
                o[i] = rand(rng, d.an)
            end
        else
            o[i] = rand(rng, d.n)
        end
    end
    return TrackObs(SVector(o), detect)
end

function POMDPs.pdf(d::BeamDist,   o::TrackObs)
    p = 1.0
    for i in 1:length(o.obs)
        if i == d.abeam
            p *= POMDPs.pdf(d.an, o.obs[i])
        else
            p *= POMDPs.pdf(d.n, o.obs[i])
        end
    end

    p *= POMDPs.pdf(d.p_detect, o.detect)
    return p
end

# POMDPs.pdf(d::BeamDist,  o::TrackObs) = pdf(d, o.obs)

function active_beam(rel_pos::Vec2)
    angle = atan(rel_pos[2], rel_pos[1])
    while angle <= 0.0
        angle += 2*pi
    end
    bm = ceil(Int, 8*angle/(2*pi))
    return clamp(bm, 1, 8)
end

function POMDPs.observation(p::VDPTrackPOMDP, a::TrackAction, sp::TrackState)
    rel_pos = sp.target[a]
    dist = norm(rel_pos)
    abeam = active_beam(rel_pos)
    
    
    an_detect = Normal(dist, p.active_meas_std)
    
    an = Normal(dist, p.meas_std)
    p_detect = Bernoulli(p.p_detect[a])
    
    n = Normal(1.0, p.meas_std)

    BeamDist(abeam, an_detect, an, n, p_detect)
end

# POMDPs.observation(p::VDPTrackPOMDP, a::TrackAction, sp::TrackState) = observation(p, a, sp)

include("rk4.jl")
# include("barriers.jl")
include("initial.jl")
# include("discretized.jl")
include("visualization.jl")
# include("heuristics.jl")

function ModelTools.gbmdp_handle_terminal(pomdp::VDPTrackPOMDP, updater::Updater, b::ParticleCollection, s, a, rng)
    return ParticleCollection([s])
end

end # module
