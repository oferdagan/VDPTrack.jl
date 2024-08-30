function rk4step(p::VDPTrackMDP, pos::Vec2, obj)
    h = p.dt
    k1 = vdp_dynamics(p.mu[obj], pos)
    k2 = vdp_dynamics(p.mu[obj], pos+h/2*k1)
    k3 = vdp_dynamics(p.mu[obj], pos+h/2*k2)
    k4 = vdp_dynamics(p.mu[obj], pos+h*k3)
    return pos + h/6*(k1+2*k2+2*k3+k4)
end

function vdp_dynamics(mu::Float64, pos::Vec2)
    x = pos[1]
    y = pos[2]
    return SVector(mu*(x-x^3/3-y), x/mu)
end
