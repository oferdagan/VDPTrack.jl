using Random

struct VDPInitDist end
sampletype(::Type{VDPInitDist}) = TrackState
function rand(rng::AbstractRNG, d::VDPInitDist)
    # return TagState([0.0, 0.0], 8.0*rand(rng, 2)-4.0)
    targ = []
        for obj in range(1,3)
            push!(targ, 1.0*rand(rng, 2) .- 1.0)
        end
    return TrackState(targ, 0)

end

POMDPs.initialstate(::VDPTrackProblem) = VDPInitDist()
