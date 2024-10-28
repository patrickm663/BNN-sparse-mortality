### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 385ae66e-77ff-11ef-2d77-95868f896088
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
end

# ╔═╡ 9a8bcc40-a57d-499f-b24e-3460d46b6320
using AdvancedHMC, CUDA, Lux, LinearAlgebra, ForwardDiff, Tracker, Random, LogDensityProblems, Plots, StatsPlots

# ╔═╡ 93832dbf-3a1a-43ae-82ce-a9fa7263d94b
using Distributions: logpdf, InverseGamma, Normal

# ╔═╡ 2e090494-0356-4fa6-a077-1dd0cd518684
using Bijectors: invlink, logpdf_with_trans

# ╔═╡ 9b76a1a0-6d95-4ed7-9f43-43a70e182f7c
begin
	gdev = gpu_device()
	cdev = cpu_device()
	CUDA.allowscalar(false) 
end

# ╔═╡ d0a606fa-96cc-428a-b980-79e2f6b4e28c
rng = Xoshiro(321)

# ╔═╡ 1b8e377c-e3e4-48f6-9721-104ebfefab00
function invlink_gdemo(θ)
    s = invlink(InverseGamma(2, 3), θ[1])
    m = θ[2]
	
	return [s, m]
end

# ╔═╡ 4c3c92a5-cd03-4adf-86ea-aa2477e73732


# ╔═╡ 5f31f238-9130-4794-9e87-1d178d0aadf8
function ℓπ_gdemo(θ)
    s, m = invlink_gdemo(θ)
    logprior =
        logpdf_with_trans(InverseGamma(2, 3), s, true) + logpdf(Normal(0, sqrt(s)), m)
    loglikelihood = logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
    return logprior + loglikelihood
end

# ╔═╡ fc61c4e1-a77b-427f-b611-7102d248311f
LogDensityProblems.dimension(::typeof(ℓπ_gdemo)) = 2

# ╔═╡ 85244f8e-4abe-4efb-b1ad-bee9929d2725
LogDensityProblems.logdensity(::typeof(ℓπ_gdemo), θ) = ℓπ_gdemo(θ)

# ╔═╡ e578cd77-c681-49ce-b26d-7e9351bf8cab
LogDensityProblems.capabilities(::Type{typeof(ℓπ_gdemo)}) =
    LogDensityProblems.LogDensityOrder{0}()

# ╔═╡ e802ab62-a738-4cfe-bb3e-2699aeb6c68a
n_samples = 5_000

# ╔═╡ 6eba5494-971e-4254-850f-603867fca618
n_adapts = 1_000

# ╔═╡ df3ce855-af20-4c22-bf88-fc2ce606f43c
θ_init = randn(rng, 2)

# ╔═╡ 9cfae539-dead-4f44-b923-9b4bf6ee3b12
metric = DiagEuclideanMetric(Float32, size(θ_init))

# ╔═╡ 5cab12b8-1874-4cfb-afb6-4b7de0b78508
h = Hamiltonian(metric, ℓπ_gdemo, ForwardDiff)

# ╔═╡ 922979da-8bba-46a7-aa59-673ac75ac48b
integrator = Leapfrog(find_good_stepsize(h, θ_init))

# ╔═╡ dcb012a7-250e-4c91-8b08-0dfff1731344
κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))

# ╔═╡ c423fb73-deee-45eb-a62e-f0d33b51fa40
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.9, integrator))

# ╔═╡ 5771c3d0-adfe-45bb-b434-16b1243752df
@time samples, stats = sample(
	rng,
	h,
	κ,
	θ_init,
	n_samples,
	adaptor,
	n_adapts;
	progress = true,
	verbose = true,
)

# ╔═╡ d9cbe511-b17a-4318-98c9-54344f768218
density(hcat(samples...)[1, :], label="Var 1")

# ╔═╡ f67cef3f-3a5f-4ce9-8980-6b96c961e364
density(hcat(samples...)[2, :], label="Var 2")

# ╔═╡ Cell order:
# ╠═385ae66e-77ff-11ef-2d77-95868f896088
# ╠═9a8bcc40-a57d-499f-b24e-3460d46b6320
# ╠═93832dbf-3a1a-43ae-82ce-a9fa7263d94b
# ╠═2e090494-0356-4fa6-a077-1dd0cd518684
# ╠═9b76a1a0-6d95-4ed7-9f43-43a70e182f7c
# ╠═d0a606fa-96cc-428a-b980-79e2f6b4e28c
# ╠═1b8e377c-e3e4-48f6-9721-104ebfefab00
# ╠═4c3c92a5-cd03-4adf-86ea-aa2477e73732
# ╠═5f31f238-9130-4794-9e87-1d178d0aadf8
# ╠═fc61c4e1-a77b-427f-b611-7102d248311f
# ╠═85244f8e-4abe-4efb-b1ad-bee9929d2725
# ╠═e578cd77-c681-49ce-b26d-7e9351bf8cab
# ╠═e802ab62-a738-4cfe-bb3e-2699aeb6c68a
# ╠═6eba5494-971e-4254-850f-603867fca618
# ╠═df3ce855-af20-4c22-bf88-fc2ce606f43c
# ╠═9cfae539-dead-4f44-b923-9b4bf6ee3b12
# ╠═5cab12b8-1874-4cfb-afb6-4b7de0b78508
# ╠═922979da-8bba-46a7-aa59-673ac75ac48b
# ╠═dcb012a7-250e-4c91-8b08-0dfff1731344
# ╠═c423fb73-deee-45eb-a62e-f0d33b51fa40
# ╠═5771c3d0-adfe-45bb-b434-16b1243752df
# ╠═d9cbe511-b17a-4318-98c9-54344f768218
# ╠═f67cef3f-3a5f-4ce9-8980-6b96c961e364
