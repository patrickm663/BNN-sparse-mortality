# Source: https://docs.sciml.ai/Overview/stable/showcase/bayesian_neural_ode/

# SciML Libraries
using SciMLSensitivity, DifferentialEquations

# ML Tools
using Lux, Zygote

# External Tools
using Random, Plots, AdvancedHMC, MCMCChains, StatsPlots, ComponentArrays

include("lee_carter.jl")

#kappa_std = (std(kappa_males), std(kappa_females))
#kappa_males = kappa_males[1:end] ./ kappa_std[1]
#kappa_females = kappa_females[1:end] ./ kappa_std[2]

X_train_square_log_males = X_train_square_log_males[1:10:end, [1, 7, 15, 16, 22, 38, 48, 51]]

u0 = Array(X_train_square_log_males[:, 1]) #[kappa_males[1]; kappa_females[1]]
datasize = 51#size(X_train_square_log_males)[2]#length(kappa_males)
tspan = (0.0, (datasize-1) |> f64)
tsteps = [1, 7, 15, 16, 22, 38, 48, 51] .- 1.0f0
long_tsteps = range(tspan[1], 80.0, length = Int(2*(80.0-tspan[1]+1)))
long_tspan = (0.0, long_tsteps[end] |> f64)
#ode_data = Array(hcat(kappa_males, kappa_females)')
ode_data = Array(X_train_square_log_males)

dudt2 = Lux.Chain(Lux.Dense(size(ode_data)[1], 64, tanh), Dense(64 => 64, tanh), Lux.Dense(64, size(ode_data)[1]))

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
_st = st
function neuralodefunc(u, p, t)
    dudt2(u, p, _st)[1]
end
function prob_neuralode(u0, p)
    prob = ODEProblem(neuralodefunc, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tsteps)
end

function forecast_prob_neuralode(u0, p)
    prob_ = ODEProblem(neuralodefunc, u0, long_tspan, p)
    sol_ = solve(prob_, Tsit5(), saveat = long_tsteps) 
end

p = ComponentArray{Float64}(p)
_p = p

function predict_neuralode(p)
    p = p isa ComponentArray ? p : convert(typeof(_p), p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

function forecast_neuralode(p)
    p__ = p isa ComponentArray ? p : convert(typeof(_p),p)
    Array(forecast_prob_neuralode(u0, p__))
end

l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)
function dldθ(θ)
    x, lambda = Zygote.pullback(l, θ)
    grad = first(lambda(1))
    return x, grad
end

metric = DiagEuclideanMetric(length(p))
h = Hamiltonian(metric, l, dldθ)

integrator = Leapfrog(find_good_stepsize(h, p))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.9, integrator))
N_samples = 1_000
samples, stats = sample(h, kernel, p, N_samples, adaptor, N_samples; progress = true)

samples = hcat(samples...)
samples_reduced = samples[1:5, :]
samples_reshape = reshape(samples_reduced, (N_samples, 5, 1))
BNN_chains = Chains(samples_reshape)
plot(BNN_chains)

autocorplot(BNN_chains)

pl1 = scatter(tsteps .+ 1950.0, ode_data[1, :], color = :red, label = "Data: Age 0", xlabel = "t",
             title = "BNN: Mortality")
scatter!(tsteps .+ 1950.0, ode_data[2, :], color = :blue, label = "Data: Age 1")
for k in 1:500
    resol = forecast_neuralode(samples[:, 100:end][:, rand(1:(N_samples-100))])
    plot!(long_tsteps .+ 1950.0, resol[1, :], alpha = 0.04, color = :red, label = "")
    plot!(long_tsteps .+ 1950.0, resol[2, :], alpha = 0.04, color = :blue, label = "")
end

losses = map(x -> loss_neuralode(x)[1], eachcol(samples))
idx = findmin(losses)[2]
prediction = forecast_neuralode(samples[:, idx])
plot!(long_tsteps .+ 1950.0, prediction[1, :], color = :black, w = 2, label = "")
plot!(long_tsteps .+ 1950.0, prediction[2, :], color = :black, w = 2, label = "MAP Estimate")


