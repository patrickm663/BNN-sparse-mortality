### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 73791d5e-691d-11ef-1cca-35c52ceada8f
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
end

# ╔═╡ a8c535ac-8ccf-4718-8214-c230adcf0b75
begin
	using Lux, Tracker, Optimisers, Functors
	using DataFrames, CSV, HMD, Plots, StatsPlots
	using Turing, Distributions
	using Random, LinearAlgebra, ComponentArrays
end

# ╔═╡ c87201c0-fbb6-435b-b956-fd62279f6697
md"""
Load the required packages.
"""

# ╔═╡ 0c25f42a-513e-4369-96c0-2b2bd0ac6a04


# ╔═╡ 07b39487-13d5-4f13-9b86-7f9f8e9b920c
md"""
Set a random seed for consistency.
"""

# ╔═╡ f61d2994-d4c7-4cbd-bbb1-09f419c2b62e
begin
	rng = Random.default_rng()
	Random.seed!(rng, 456789)
end

# ╔═╡ cb80ed75-b2ce-4fae-8346-93c68c2aeb7f
md"""
Load the full USA mortality dataset, capping the age range at 100 and year range at 1950.
"""

# ╔═╡ 657a1395-6196-420d-86d4-079cc783e827
begin
	USA_MX_raw = DataFrame(CSV.File("USA_Mx_1x1.csv"))
	
	# Get 1950+ and ≤ 100
	USA_MX_raw = USA_MX_raw[(USA_MX_raw.Year .≥ 1950) .&& (USA_MX_raw.Age .≤ 100), :]

	# For Lee-Carter
	USA_MX_square_log_males = log.(HMD.transform(USA_MX_raw, :Male)[:, 2:end])
	USA_MX_square_log_females = log.(HMD.transform(USA_MX_raw, :Female)[:, 2:end])

	# For NN
	USA_MX_matrix = Matrix{Float64}(undef, 101*(2021-1950+1)*2, 4)

	global c = 0
	for i in 0:1
		for j in 1950:2021
			for k in 0:100
				global c += 1
				USA_MX_matrix[c, 1] = j
				USA_MX_matrix[c, 2] = k
				USA_MX_matrix[c, 3] = i
				# Male = 1
				if i == 1
					male_MX = log(USA_MX_raw[(USA_MX_raw.Year .== j) .&& (USA_MX_raw.Age .== k), :Male][1])
					USA_MX_matrix[c, 4] = male_MX
				# Female = 0
				else
					female_MX = log(USA_MX_raw[(USA_MX_raw.Year .== j) .&& (USA_MX_raw.Age .== k), :Female][1])
					USA_MX_matrix[c, 4] = female_MX
				end
			end
		end
	end

	USA_MX_matrix
end

# ╔═╡ a92d0078-8652-494a-88df-144577f756a5
md"""
Create the Z-transformation scales. We can apply this because we know the minimum and maximum age and years of the observation -- regardless of missing data in between.
"""

# ╔═╡ 37ed48f0-2be6-47e1-8725-507f1d0b3ee1
begin
	# Training/seen data is 1950-2000
	age_range = 0:100
	year_range = 1950:2000
	
	age_mu, age_sigma = (mean(age_range), std(age_range))
	year_mu, year_sigma = (mean(year_range), std(year_range))
end

# ╔═╡ c33ccbfb-24bf-462f-8d94-67a961decd79
rescale(x, μ, σ) = (x .* σ) .+ μ

# ╔═╡ d7107f3d-0d5c-476f-b479-fe295828f7b0
md"""
Scale the training set and create a 'grid' for the Lee-Carter model to process (on all the data). TODO: explore Lee-Carter with missing values.
"""

# ╔═╡ 9a9e8419-c125-433d-8dd9-c2c494af8f3a
begin
	# First get 'full' X_train and y_train for NN
	X_train = USA_MX_matrix[USA_MX_matrix[:, 1] .≤ year_range[end], 1:3]
	X_train[:, 1] = (X_train[:, 1] .- year_mu) ./ year_sigma
	X_train[:, 2] = (X_train[:, 2] .- age_mu) ./ age_sigma
	y_train = USA_MX_matrix[USA_MX_matrix[:, 1] .≤ year_range[end], 4]

	# Get LC train set
	X_train_square_log_males = Matrix(USA_MX_square_log_males[:, year_range .- (year_range[1] - 1)])
	X_train_square_log_females = Matrix(USA_MX_square_log_females[:, year_range .- (year_range[1] - 1)])

	# Test/unseen is everything after 2000 for NN
	X_test = USA_MX_matrix[year_range[end] .< USA_MX_matrix[:, 1], 1:3]
	X_test[:, 1] = (X_test[:, 1] .- year_mu) ./ year_sigma
	X_test[:, 2] = (X_test[:, 2] .- age_mu) ./ age_sigma
	y_test = USA_MX_matrix[year_range[end] .< USA_MX_matrix[:, 1], 4]

	# Get LC test set
	X_test_square_log_males = Matrix(USA_MX_square_log_males[:, (2001:2021) .- (year_range[1] - 1)])
	X_test_square_log_females = Matrix(USA_MX_square_log_females[:, (2001:2021) .- (year_range[1] - 1)])
end

# ╔═╡ 66063226-138f-4416-9aa5-8e7ad6fa004e
md"""
Create a function to perform SVD to construct the Lee-Carter model.
"""

# ╔═╡ 540ba303-8602-4d96-97a4-e67dd591f47b
function lee_carter(d)
	log_A = Matrix(d)
	αₓ = mean(log_A; dims=2)
	log_A_std = log_A .- αₓ
	U, λ, Vt = svd(log_A_std; full=true)
	bₓ = U[:, 1]
	kₜ = (λ[1] .* Vt[:, 1])
	log_LC = αₓ .+ bₓ * kₜ'
	
	return log_LC, αₓ, bₓ, kₜ
end

# ╔═╡ 4322bdb1-2889-40fe-b9ad-7b4398ebce5a
md"""
Create a Male and Female version. This is because Lee-Carter does not have a way to handle features other than age and year.
"""

# ╔═╡ d31f81fd-0908-48d5-aae6-d541f0b0354c
log_LC_males, alpha_males, beta_males, kappa_males = lee_carter(X_train_square_log_males)

# ╔═╡ b3ec6cfc-95d8-438e-9b8e-3194d00dd475
log_LC_females, alpha_females, beta_females, kappa_females = lee_carter(X_train_square_log_females)

# ╔═╡ f2061baa-90da-41fa-9c9c-475882c9f475
log_LC_males

# ╔═╡ 99e91545-ac94-4fe9-8429-4087eeb6d080
begin
	plot(xlab="Age", ylab="log(μₓ)", title="Lee-Carter:\n 1950:2000 (In-sample)", legend=:bottomright)
	for i ∈ 1950:10:2000
		scatter!(0:100, X_train_square_log_males[:, i-1950+1], label="Observation: $(i)", markersize=1)
		plot!(0:100, log_LC_males[:, i-1950+1], label="Lee-Carter: $(i)", width=2)
	end
	plot!()
end

# ╔═╡ 894c18c2-a145-400e-9b5b-c37af5be0282
print("In-sample MSE (Males): ", mean(mean((X_train_square_log_males .- log_LC_males) .^ 2; dims=2), dims=1)[1])

# ╔═╡ d9ef687e-914b-4ea7-96b3-7c872a7da035
sum(kappa_males)

# ╔═╡ be6f3755-72c9-41d7-aad1-cc53a4addc7e
sum(beta_males)

# ╔═╡ ea534ede-adde-4e69-aa28-45dc7d8b69f8
plot(kappa_males)

# ╔═╡ b2ed19f8-9bd8-4965-b518-369c07e3c633
md"""
Create a function to forecast Kappa t years into the future, given the existing values of Kappa.
"""

# ╔═╡ 45559a00-b2c8-477d-b299-febaa68af5fb
function lee_carter_sigma_mle(kappa, t)
	μ = (kappa[end] - kappa[1]) ./ (length(kappa)-1)
	σ = 0.0
	for i in 1:(t-1)
		σ += (kappa[t+1] - kappa[t] - μ)^2
	end
	return μ, σ/(t-1)
end

# ╔═╡ 48404090-3f4e-4659-9613-79796dc1b977
function lee_carter_forecast(kappa, t, N)
	# Return a matrix with N random samples of length t
	
	kappa_proj = Matrix{Float64}(undef, t, N)
	μ, σ = lee_carter_sigma_mle(kappa, t)

	global ϵ = 0

	for s in 1:N
		ϵ = ϵ + 1
		for pr in 1:t
			if pr == 1
				kappa_proj[pr, s] = kappa[end]
			else
				kappa_proj[pr, s] = kappa_proj[pr-1, s] + rand(Xoshiro(1456789+ϵ), Normal(μ, sqrt(σ)))
			end
		end
	end

	return kappa_proj
end

# ╔═╡ 9a2f1a56-8747-4055-aa30-a289a3ee2432
begin
	plot(xlab="Age", ylab="log(μₓ)", title="Lee-Carter\n 2001:2021 (Out-sample)", legend=:bottomright)
	for i ∈ 2001:5:2021
		for j ∈ 1:100
			kappa_males_forecast = lee_carter_forecast(kappa_males, 2021-2001+1, 1)
			log_LC_test_males = alpha_males .+ beta_males .* kappa_males_forecast'
			plot!(0:100, log_LC_test_males[:, i-2001+1], label="", width=1, color=:grey, alpha=0.3)
		end
		scatter!(0:100, X_test_square_log_males[:, i-2001+1], label="Observation: $(i)")
	end
	plot!()
end

# ╔═╡ de4ec6fd-7d3d-4871-8611-9824c8cabf72
begin
	plot(1950:2021, vcat([kappa_males, lee_carter_forecast(kappa_males, 21, 1)]...), title="Kappa Forecast\n 21 Years Ahead", label="", xlab="Years", ylab="log(μ)")
	vline!([2000], label="Seen/Unseen Cut-off", linestyle=:dash)
end

# ╔═╡ a267e96d-d460-4204-a4a9-e4aa7dc29c1e
md"""
This function acts as the "main" entry point. It constructs a traditional FNN and trains it using Adam optimiser, then it constructs a BNN and trains it using the NUTS inference methods. After training, the age plot is constructed. The Age Plot function does posterior sampling on the parameters of the BNN to produce prediction intervals.
"""

# ╔═╡ dafa4fa6-8616-4c2e-8f27-c71e29fbe50d
function append_BNN_params(N, full_param_length, random_param_length, ch, initial_params, bkwd)
	θ_full = zeros(N, full_param_length)
	
	for i ∈ 1:N
		if bkwd == true
			θ_full[i, :] = vcat([initial_params[1:(end-random_param_length)], MCMCChains.group(ch, :parameters).value[i, :, :]]...)
		else
			θ_full[i, :] = vcat([MCMCChains.group(ch, :parameters).value[i, :, :], initial_params[(random_param_length+1):end]]...)
		end
	end
	
	return θ_full
end

# ╔═╡ 35a6b42c-1704-4c0e-b946-75f47b8dd70b
function vector_to_parameters_(ps_new::AbstractVector, ps::NamedTuple)
	@assert length(ps_new) == Lux.parameterlength(ps)
	i = 1
	function get_ps(x)
		z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
		i += length(x)
		return z
	end
	return fmap(get_ps, ps)
end

# ╔═╡ 000d6486-ba8a-4260-9077-ef9f1d1fe34f
nn_forward(x, θ, nn, ps, st) = vec(first(nn(x, vector_to_parameters_(θ, ps), st)))

# ╔═╡ 6dfb890f-853f-4634-8264-c28ee9d62354
md"""
Loops over different percentages (1%, 5%, etc.) of the training data and call the "BNN" function.
"""

# ╔═╡ a9fc086b-61d9-4ced-9dca-59a53fa83e39
device = gpu_device()

# ╔═╡ 576550d1-fb64-4edc-8610-0e8bd745711d
md"""
The Age Plot function is called internally in the above BNN function. It produces 100 000 posterior estimates for plotting.
"""

# ╔═╡ cef77b70-d171-4254-9fe7-0a6b0ce8ede3
function lee_carter_full_forecast(sample_N)
	lc_pred_males_samples = lee_carter_forecast(kappa_males, 21, sample_N)
	
	kappa_males_forecast_mean = mean(lc_pred_males_samples; dims=2)
	log_LC_test_males_mean = vcat((alpha_males .+ beta_males .* kappa_males_forecast_mean')...)
	
	kappa_males_forecast_l05 = quantile.(eachrow(lc_pred_males_samples), 0.95)
	log_LC_test_males_l05 = vcat((alpha_males .+ beta_males .* kappa_males_forecast_l05')...)

	kappa_males_forecast_med = quantile.(eachrow(lc_pred_males_samples), 0.5)
	log_LC_test_males_med = vcat((alpha_males .+ beta_males .* kappa_males_forecast_med')...)
	
	kappa_males_forecast_u95 = quantile.(eachrow(lc_pred_males_samples), 0.05)
	log_LC_test_males_u95 = vcat((alpha_males .+ beta_males .* kappa_males_forecast_u95')...)

	lc_pred_females_samples = lee_carter_forecast(kappa_females, 21, sample_N)
	
	kappa_females_forecast_mean = mean(lc_pred_females_samples; dims=2)
	log_LC_test_females_mean = vcat((alpha_females .+ beta_females .* kappa_females_forecast_mean')...)
	
	kappa_females_forecast_l05 = quantile.(eachrow(lc_pred_females_samples), 0.95)
	log_LC_test_females_l05 = vcat((alpha_females .+ beta_females .* kappa_females_forecast_l05')...)

	kappa_females_forecast_med = quantile.(eachrow(lc_pred_females_samples), 0.5)
	log_LC_test_females_med = vcat((alpha_females .+ beta_females .* kappa_females_forecast_med')...)
	
	kappa_females_forecast_u95 = quantile.(eachrow(lc_pred_females_samples), 0.05)
	log_LC_test_females_u95 = vcat((alpha_females .+ beta_females .* kappa_females_forecast_u95')...)

	return log_LC_test_males_mean, log_LC_test_males_l05, log_LC_test_males_u95, log_LC_test_females_mean, log_LC_test_females_l05, log_LC_test_females_u95, log_LC_test_males_med, log_LC_test_females_med
end

# ╔═╡ d1208799-3cf9-4f26-8a4a-b88ae2a03108
function pred_interval_score(X_, y_, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate_, training_state, θ_samples, θ_for_MAP)
	
	sample_N = max(N, 10_000)
	nn_pred_samples = Matrix{Float64}(undef, sample_N, size(X_)[1])

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_', θ_samples[i, :], nn, ps, st)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
	nn_pred_MAP =  nn_forward(X_', θ_for_MAP[idx, :], nn, ps, st)

	# Apply PICP and MPIW
	PICP(lb, ub, dp) = sum(lb .≤ dp .≤ ub) / size(dp)[1]
	MPIW(lb, ub) = sum(ub .- lb) / size(lb)[1]

	pred_error_df = DataFrame(:Score => ["PICP", "MPIW", "MSE_mean", "MSE_median"], :BNN => zeros(4), :LC => zeros(4))
	
	pred_error_df[1, :BNN] = PICP(nn_pred_l05, nn_pred_u95, y_)
	pred_error_df[2, :BNN] = MPIW(nn_pred_l05, nn_pred_u95)
	pred_error_df[3, :BNN] = mean((nn_pred_mean .- y_) .^ 2)
	pred_error_df[4, :BNN] = mean((nn_pred_median .- y_) .^ 2)

	if training_state == "TEST"
			
		log_LC_test_males_mean, log_LC_test_males_l05, log_LC_test_males_u95, log_LC_test_females_mean, log_LC_test_females_l05, log_LC_test_females_u95, log_LC_test_males_med, log_LC_test_females_med = lee_carter_full_forecast(sample_N)

		log_LC_test_l05 = vcat([log_LC_test_females_l05, log_LC_test_males_l05]...)
		log_LC_test_u95 = vcat([log_LC_test_females_u95, log_LC_test_males_u95]...)
		log_LC_test_mean = vcat([log_LC_test_females_mean, log_LC_test_males_mean]...)
		log_LC_test_med = vcat([log_LC_test_females_med, log_LC_test_males_med]...)

		pred_error_df[1, :LC] = PICP(log_LC_test_l05, log_LC_test_u95, USA_MX_matrix[(USA_MX_matrix[:, 1] .> 2000), 4])
		pred_error_df[2, :LC] = MPIW(log_LC_test_l05, log_LC_test_u95)

		pred_error_df[3, :LC] = mean((log_LC_test_mean .- y_) .^ 2)
		pred_error_df[4, :LC] = mean((log_LC_test_med .- y_) .^ 2)
	end

	CSV.write("results/pred_error_MSE_$(size_of_data_split)-$(N)-$(training_state).csv", pred_error_df)
	
end

# ╔═╡ 0e8cc4c5-2ad2-493d-94e1-e312518d6ddd
function sample_BNN_parameters(ch, N, size_of_data_split; pseudo_bnn=Dict("init_params" => 0, "flag" => false, "random_params" => 0, "full_params" => 0), bkwd)
	sample_N = max(N, 10_000)
	posterior_samples = sample(Xoshiro(1456789), ch, sample_N)
	
	if pseudo_bnn["flag"] == true
		θ_samples = append_BNN_params(sample_N, pseudo_bnn["full_params"], pseudo_bnn["random_params"], posterior_samples, pseudo_bnn["init_params"], bkwd)
		θ_for_MAP = append_BNN_params(N, pseudo_bnn["full_params"], pseudo_bnn["random_params"], ch, pseudo_bnn["init_params"], bkwd)
	else
		θ_samples = MCMCChains.group(posterior_samples, :parameters).value
		θ_for_MAP = MCMCChains.group(ch, :parameters).value
	end

	return θ_samples, θ_for_MAP
end

# ╔═╡ e00a61f5-b488-48c2-a3b5-79f6f1f44081
function age_plot(year, gender, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate_, θ_samples, θ_for_MAP; pseudo="")
	if year ≤ 2000
		X_test_ = X_train[(X_train[:, 1] .== (year .- year_mu) ./ year_sigma) .&& (X_train[:, 3] .== gender), :]
		samples_one_p = USA_MX_matrix[USA_MX_matrix[:, 1] .≤ 2000, :][perc, :]
		samples_one_p = samples_one_p[(samples_one_p[:, 1] .== year) .&& (samples_one_p[:, 3] .== gender), :]
	else
		X_test_ = X_test[(X_test[:, 1] .== (year .- year_mu) ./ year_sigma) .&& (X_test[:, 3] .== gender), :]
	end

	sample_N = max(N, 10_000)
	nn_pred_samples = Matrix{Float64}(undef, sample_N, size(X_test_)[1])

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_test_', θ_samples[i, :], nn, ps, st)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
	nn_pred_MAP =  nn_forward(X_test_', θ_for_MAP[idx, :], nn, ps, st)

	# Apply PICP and MPIW
	PICP(lb, ub, dp) = sum(lb .≤ dp .≤ ub) / size(dp)[1]
	MPIW(lb, ub) = sum(ub .- lb) / size(lb)[1]

	fnn_pred = Lux.apply(tstate_.model, X_test_', tstate_.parameters, tstate_.states)[1]

	if year > 2000
		if gender == 1
			
			lc_pred_males_samples = lee_carter_forecast(kappa_males, year-2001+1, sample_N)[end, :]
			
			kappa_males_forecast_mean = mean(lc_pred_males_samples; dims=1)
			log_LC_test_males_mean = alpha_males .+ beta_males .* kappa_males_forecast_mean'
			
			kappa_males_forecast_l05 = quantile.(eachcol(lc_pred_males_samples), 0.95)
			log_LC_test_males_l05 = alpha_males .+ beta_males .* kappa_males_forecast_l05'
			
			kappa_males_forecast_u95 = quantile.(eachcol(lc_pred_males_samples), 0.05)
			log_LC_test_males_u95 = (alpha_males .+ beta_males .* kappa_males_forecast_u95')

		else
			lc_pred_females_samples = lee_carter_forecast(kappa_females, year-2001+1, sample_N)[end, :]
			
			kappa_females_forecast_mean = mean(lc_pred_females_samples; dims=1)
			log_LC_test_females_mean = alpha_females .+ beta_females .* kappa_females_forecast_mean'
			
			kappa_females_forecast_l05 = quantile.(eachcol(lc_pred_females_samples), 0.95)
			log_LC_test_females_l05 = alpha_females .+ beta_females .* kappa_females_forecast_l05'
			
			kappa_females_forecast_u95 = quantile.(eachcol(lc_pred_females_samples), 0.05)
			log_LC_test_females_u95 = alpha_females .+ beta_females .* kappa_females_forecast_u95'
			
		end
	end
	
	p_ = begin
		# Plot front-matter
		if year ≤ 2000
			if gender == 1
				title_ = "USA Mortality on $(size_of_data_split) of Data \n In-sample: $(year); Males"
			else
				title_ = "USA Mortality on $(size_of_data_split) of Data \n In-sample: $(year); Females"
			end
		else
			if gender == 1
				title_ = "USA Mortality on $(size_of_data_split) of Data \n Out-sample: $(year); Males"
			else
				title_ = "USA Mortality on $(size_of_data_split) of Data \n Out-sample: $(year); Females"
			end
		end
		plot(title=title_, xlab="Age", ylab="log(μ)", label="", legend=:outertopright)

		# Plot Lee-Carter
		if year ≤ 2000
			if gender == 1
				plot!(0:100, log_LC_males[:, year-1950+1], label="Lee-Carter", width=2, color=:brown, style=:dash)
			else
				plot!(0:100, log_LC_females[:, year-1950+1], label="Lee-Carter", width=2, color=:brown, style=:dash)
			end
		else
			if gender == 1
				plot!(0:100, log_LC_test_males_l05, fillrange = log_LC_test_males_u95, color=:brown, width=.1, alpha=.4, label="")
				plot!(0:100, log_LC_test_males_mean, label="", color=:brown, width=2, style=:dash)
			else
				plot!(0:100, log_LC_test_females_l05, fillrange = log_LC_test_females_u95, color=:brown, width=.1, alpha=.4, label="")
				plot!(0:100, log_LC_test_females_mean, label="", color=:brown, width=2, style=:dash)
			end
		end
		
		plot!(0:100, nn_pred_l05, fillrange = nn_pred_u95, label="BNN: 5-95% CI", color=:blue, width=.1, alpha=.3)
	
		# Plot BNN mean
		plot!(0:100, nn_pred_mean, label="BNN: Mean", color=:blue, width=2)

		# Plot BNN median
		plot!(0:100, nn_pred_median, label="BNN: Median", color=:blue, width=2, style=:dashdot)

		# Plot FNN
		plot!(0:100, fnn_pred', label="FNN", color=:green, width=2)
		
		# Plot BNN MAP
		plot!(0:100, nn_pred_MAP, label="BNN: MAP", color=:blue, width=2, style=:dot)
		
		# Plot Observations
		if year ≤ 2000
			scatter!(samples_one_p[:, 2], samples_one_p[:, 4], label="Observed Samples")
		end
		
		scatter!(0:100, USA_MX_matrix[(USA_MX_matrix[:, 1] .== year) .&& (USA_MX_matrix[:, 3] .== gender), 4], label="Full Underlying Data", color=:black, markershape=:circle, markersize=1.5, ylim=(-10.0, -0.5))

		plot!()
	end

	p_

	savefig(p_, "results/$(year)-$(gender)-$(size_of_data_split)-$(N)-BNN$(pseudo).png")
	
	return p_
end

# ╔═╡ 1e172ba4-60a7-4011-b756-8bb47b1d8a92
function BNN(Xs, ys, N, perc, size_of_data_split; sampling_algorithm="NUTS")
	half_N = Int(ceil(N/2))
	
	# Construct a neural network using Lux
	nn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	# Initialize the model weights and state
	ps, st = Lux.setup(rng, nn)

	fnn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	ps_fnn, st_fnn = Lux.setup(rng, fnn)

	opt = Adam(0.03f0)

	loss_function = MSELoss()

	tstate = Training.TrainState(fnn, ps_fnn, st_fnn, opt)

	vjp_rule = AutoTracker()

	function train_FNN(tstate::Training.TrainState, vjp, data, epochs)
		losses = []
	    for epoch in 1:epochs
	        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
	      	push!(losses, loss)
	    end
	    return tstate, losses
	end
	
	t_fnn = time()
	tstate, losses_ = train_FNN(tstate, vjp_rule, (Xs', Matrix(ys')), max(3N, 5_000))
	dt_fnn = time() - t_fnn

	CSV.write("results/FNN_losses_$(size_of_data_split)_$(N).csv", DataFrame(Matrix(losses_'), :auto))
	
	savefig(plot(half_N:length(losses_), losses_[half_N:end], title="FNN Training Losses $(size_of_data_split)", xlab="Epochs", ylab="MSE"), "results/FNN_losses_plot_$(size_of_data_split)_$(N).png")
	
	FNN_forward(X_) = Lux.apply(tstate.model, X_, tstate.parameters, tstate.states)[1]

	# Create a regularization term and a Gaussian prior variance term.
	alpha = 0.8
	sig = 1.0 / sqrt(alpha)

	function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
	    @assert length(ps_new) == Lux.parameterlength(ps)
	    i = 1
	    function get_ps(x)
	        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
	        i += length(x)
	        return z
	    end
	    return fmap(get_ps, ps)
	end

	# Specify the probabilistic model.
	@model function bayes_nn(xs, y, ::Type{T} = Float64) where {T}

		# HalfNormal prior on σ
		σ ~ truncated(Normal(0, 1); lower=1e-9)
	
	    # Sample the parameters from a MvNormal with 0 mean and constant variance
	    nparameters = Lux.parameterlength(nn)
	    parameters ~ MvNormal(zeros(nparameters), (sig^2) .* I)
	
	    # Forward NN to make predictions
	    preds, st = nn(xs, vector_to_parameters(parameters, ps), st)
	
	    # Age-range log(μ) are each given a MvNormal likelihood with constant variance
	    y ~ MvNormal(vec(preds), (σ^2) .* I)
	
		return Nothing
	end

	t_bnn = time()
	if sampling_algorithm == "NUTS"
		ch = sample(
			Xoshiro(1456789),
			bayes_nn(Xs', ys), 
			NUTS(0.85; adtype=AutoTracker()),
			#HMCDA(200, 0.9, 0.1; adtype=AutoTracker()),
			#SGHMC(; learning_rate=1e-8, momentum_decay=0.55, adtype=AutoTracker()),
			N; 
			discard_adapt=false,
			progress=true
		)
	end
			
	#θ = zeros(100, 257)
	#for j ∈ 1:100
	#	ch = sample(
	#		Xoshiro(1456789+j+100),
	#		bayes_nn(Xs', ys), 
			#NUTS(0.55; adtype=AutoTracker()),
			#HMCDA(200, 0.9, 0.1; adtype=AutoTracker()),
	#		SGHMC(; learning_rate=1e-8, momentum_decay=0.55, adtype=AutoTracker()),
	#		N; 
	#		discard_adapt=false,
	#		progress=false		
	#	)
		
	
	#	θ_ = MCMCChains.group(ch, :parameters).value[end, :, 1]
	#	θ[j, :] = θ_
	#end
	dt_bnn = time() - t_bnn

	CSV.write("results/full_chains_$(size_of_data_split)_$(N).csv", DataFrame(MCMCChains.group(ch, :parameters).value[:, :, 1], :auto))

	# Get MAP
	_, idx = findmax(ch[:lp])
	idx = idx.I[1]

	nn_forward_(x, θ, nn, ps, st) = vec(first(nn(x, vector_to_parameters(θ, ps), st)))

	# Save summary of samples
	CSV.write("results/summary_chains_$(size_of_data_split)_$(N).csv", DataFrame(describe(ch)[1]))

	# Save trace plot
	savefig(StatsPlots.plot(ch[half_N:end, 1:80:end, :]), "results/chains_plot_$(size_of_data_split)_$(N).png")

	θ = MCMCChains.group(ch, :parameters).value

	# Save MSE
	MSE_df = DataFrame(:State => ["", "", "", "", "", "", "", ""], :MSE => zeros(8), :RMSE => zeros(8), :Time_s => zeros(8))
	MSE_df[1, :State] = "BNN_training_$(size_of_data_split)"
	MSE_df[1, :MSE] = mean((nn_forward(Xs', θ[idx, :], nn, ps, st) .- ys) .^ 2)
	MSE_df[1, :RMSE] = sqrt(MSE_df[1, :MSE])
	MSE_df[1, :Time_s] = dt_bnn
	MSE_df[2, :State] = "BNN_full_training_$(size_of_data_split)"
	MSE_df[2, :MSE] = mean((nn_forward(X_train', θ[idx, :], nn, ps, st) .- y_train) .^ 2)
	MSE_df[2, :RMSE] = sqrt(MSE_df[2, :MSE])
	MSE_df[3, :State] = "BNN_testing_$(size_of_data_split)"
	MSE_df[3, :MSE] = mean((nn_forward(X_test', θ[idx, :], nn, ps, st) .- y_test) .^ 2)
	MSE_df[3, :RMSE] = sqrt(MSE_df[3, :MSE])
	MSE_df[4, :State] = "FNN_training_$(size_of_data_split)"
	MSE_df[4, :MSE] = mean((FNN_forward(Xs') .- Matrix(ys')) .^ 2)
	MSE_df[4, :RMSE] = sqrt(MSE_df[4, :MSE])
	MSE_df[4, :Time_s] = dt_fnn
	MSE_df[5, :State] = "FNN_full_training_$(size_of_data_split)"
	MSE_df[5, :MSE] = mean((FNN_forward(X_train') .- Matrix(y_train')) .^ 2)
	MSE_df[5, :RMSE] = sqrt(MSE_df[5, :MSE])
	MSE_df[6, :State] = "FNN_testing_$(size_of_data_split)"
	MSE_df[6, :MSE] = mean((FNN_forward(X_test') .- Matrix(y_test')) .^ 2)
	MSE_df[6, :RMSE] = sqrt(MSE_df[6, :MSE])
	MSE_df[7, :State] = "LC_full_training"
	MSE_df[7, :MSE] = mean((vcat([vcat(log_LC_females...), vcat(log_LC_males...)]...) .- USA_MX_matrix[(USA_MX_matrix[:, 1] .≤ 2000), 4]) .^ 2)
	MSE_df[7, :RMSE] = sqrt(MSE_df[7, :MSE])

	log_LC_test_males_mean, log_LC_test_males_l05, log_LC_test_males_u95, log_LC_test_females_mean, log_LC_test_females_l05, log_LC_test_females_u95, log_LC_test_males_med, log_LC_test_females_med = lee_carter_full_forecast(N)

	MSE_df[8, :State] = "LC_testing"
	MSE_df[8, :MSE] = mean((vcat([vcat(log_LC_test_females_mean...), vcat(log_LC_test_males_mean...)]...) .- USA_MX_matrix[(USA_MX_matrix[:, 1] .> 2000), 4]) .^ 2)
	MSE_df[8, :RMSE] = sqrt(MSE_df[8, :MSE])

	CSV.write("results/BNN_MSE_$(size_of_data_split)_$(N).csv", MSE_df)

	θ_BNN_samples, θ_BNN_for_MAP = sample_BNN_parameters(ch, N, size_of_data_split; pseudo_bnn=Dict("init_params" => 0, "flag" => false, "random_params" => 0))

	CSV.write("results/BNN_full_posterior_samples_$(size_of_data_split)_$(N).csv", DataFrame(Matrix(θ_BNN_samples[:, :, 1]), :auto))

	# Generate and save plots
	for i ∈ 1950:10:2021
		for j ∈ 0:1
			age_plot(i, j, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, θ_BNN_samples, θ_BNN_for_MAP)
		end
	end

	pred_interval_score(X_train, y_train, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TRAIN", θ_BNN_samples, θ_BNN_for_MAP)
	pred_interval_score(X_test, y_test, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TEST", θ_BNN_samples, θ_BNN_for_MAP)

	return ch, θ, nn, ps, st, idx
end

# ╔═╡ 0158f329-3f09-4a1a-ab2f-f1595212b236
begin
	for i ∈ [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
		if i < 0.01
			percent_ = "half-%" #"$(Int(i*100))%"
		else
			percent_ = "$(Int(i*100))%"
		end
		one_p = rand(Xoshiro(1456789), Bernoulli(i), size(X_train)[1])
		X_train_one_p = X_train[one_p, :]
		y_train_one_p = y_train[one_p]
		samples_one_p_ = USA_MX_matrix[USA_MX_matrix[:, 1] .≤ 2000, :][one_p, :]
		CSV.write("results/USA_MX_$(percent_).csv", DataFrame(samples_one_p_, [:Year, :Age, :Gender, :log_Mu]))

		# 0.5% = 2 500, 1% = 5 000, 5% = 7 500, 10% = 10 000, 25% = 15 000
		if i == 0.005
			N_length = 2_500
		elseif i == 0.01
			N_length = 5_000
		elseif i == 0.05
			N_length = 7_500
		elseif i == 0.1
			N_length = 10_000
		elseif i > 0.1
			N_length = 15_000
		end

		ch_one_p, θ_one_p, nn_one_p, ps_one_p, st_one_p, idx_one_p = BNN(X_train_one_p, y_train_one_p, N_length, one_p, percent_)
		#ch_one_p, θ_one_p, nn_one_p, ps_one_p, st_one_p, idx_one_p = BNN_pseudo(X_train_one_p, y_train_one_p, N_length, one_p, percent_, 5; bkwd=false)

	end
end

# ╔═╡ f79c4cff-9664-40e3-8e6f-ca17fdacc532
function BNN_pseudo(Xs, ys, N, perc, size_of_data_split, random_params; bkwd=true)
	half_N = Int(ceil(N/2))
	
	# Construct a neural network using Lux
	nn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	# Initialize the model weights and state
	ps, st = Lux.setup(rng, nn)

	fnn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	ps_fnn, st_fnn = Lux.setup(rng, fnn)

	opt = Adam(0.03f0)

	loss_function = MSELoss()

	tstate = Training.TrainState(fnn, ps_fnn, st_fnn, opt)

	vjp_rule = AutoTracker()

	function train_FNN(tstate::Training.TrainState, vjp, data, epochs)
		losses = []
	    for epoch in 1:epochs
	        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
	      	push!(losses, loss)
	    end
	    return tstate, losses
	end
	
	t_fnn = time()
	tstate, losses_ = train_FNN(tstate, vjp_rule, (Xs', Matrix(ys')), max(3N, 5_000))
	dt_fnn = time() - t_fnn

	CSV.write("results/FNN_losses_$(size_of_data_split)_$(N).csv", DataFrame(Matrix(losses_'), :auto))
	
	savefig(plot(half_N:length(losses_), losses_[half_N:end], title="FNN Training Losses $(size_of_data_split)", xlab="Epochs", ylab="MSE"), "results/FNN_losses_plot_$(size_of_data_split)_$(N).png")
	
	FNN_forward(X_) = Lux.apply(tstate.model, X_, tstate.parameters, tstate.states)[1]

	function parameters_to_vectors(nn_param)
		init_params = zeros(Lux.parameterlength(nn_param))
		m = 0
		for i in 1:length(nn_param)
			for j in 1:2 # either weights or bias
				init_params_ = vcat(deepcopy(nn_param[i][j])...)
				for l in 1:length(init_params_)
					m = m + 1
					init_params[m] = init_params_[l]
				end
			end
		end
		return init_params
	end

	init_params = parameters_to_vectors(tstate.parameters)

	# Create a regularization term and a Gaussian prior variance term.
	alpha = 0.8
	sig = 1.0 / sqrt(alpha)

	function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
	    @assert length(ps_new) == Lux.parameterlength(ps)
	    i = 1
	    function get_ps(x)
	        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
	        i += length(x)
	        return z
	    end
	    return fmap(get_ps, ps)
	end

	# Specify the probabilistic model.
	@model function bayes_nn(xs, y, init_param, random_params, bkwd, ::Type{T} = Float64) where {T}

		# HalfNormal prior on σ
		σ ~ truncated(Normal(0, 1); lower=1e-9)
	
	    # Sample the parameters from a MvNormal with 0 mean and constant variance
	    nparameters = Lux.parameterlength(nn)
	    parameters ~ MvNormal(zeros(random_params), (sig^2) .* I)
	
	    # Forward NN to make predictions
		if bkwd == true
	    	concat_params = vcat([init_param[1:(end-random_params)], parameters]...)
		else
			concat_params = vcat([parameters, init_param[(random_params+1):end], ]...)
		end
	
		preds, st = nn(xs, vector_to_parameters(concat_params, ps), st)
	
	    # Age-range log(μ) are each given a MvNormal likelihood with constant variance
	    y ~ MvNormal(vec(preds), (σ^2) .* I)
	
		return Nothing
	end

	t_bnn = time()
	ch = sample(
		Xoshiro(1456789),
		bayes_nn(Xs', ys, init_params, random_params, bkwd), 
		NUTS(0.85; adtype=AutoTracker()),
		N; 
		discard_adapt=false,
		progress=true
	)
	
	dt_bnn = time() - t_bnn

	CSV.write("results/full_chains_$(size_of_data_split)_$(N)_pseudo.csv", DataFrame(MCMCChains.group(ch, :parameters).value[:, :, 1], :auto))

	# Get MAP
	_, idx = findmax(ch[:lp])
	idx = idx.I[1]

	nn_forward_(x, θ, nn, ps, st) = vec(first(nn(x, vector_to_parameters(θ, ps), st)))

	# Save summary of samples
	CSV.write("results/summary_chains_$(size_of_data_split)_$(N)_pseudo.csv", DataFrame(describe(ch)[1]))

	# Save trace plot
	savefig(StatsPlots.plot(ch[half_N:end, 1:end, :]), "results/chains_plot_$(size_of_data_split)_$(N)_psuedo.png")

	θ = append_BNN_params(N, Lux.parameterlength(nn), random_params, ch, init_params, bkwd)

	# Save MSE
	MSE_df = DataFrame(:State => ["", "", "", "", "", "", "", ""], :MSE => zeros(8), :RMSE => zeros(8), :Time_s => zeros(8))
	MSE_df[1, :State] = "BNN_training_$(size_of_data_split)"
	MSE_df[1, :MSE] = mean((nn_forward(Xs', θ[idx, :], nn, ps, st) .- ys) .^ 2)
	MSE_df[1, :RMSE] = sqrt(MSE_df[1, :MSE])
	MSE_df[1, :Time_s] = dt_bnn
	MSE_df[2, :State] = "BNN_full_training_$(size_of_data_split)"
	MSE_df[2, :MSE] = mean((nn_forward(X_train', θ[idx, :], nn, ps, st) .- y_train) .^ 2)
	MSE_df[2, :RMSE] = sqrt(MSE_df[2, :MSE])
	MSE_df[3, :State] = "BNN_testing_$(size_of_data_split)"
	MSE_df[3, :MSE] = mean((nn_forward(X_test', θ[idx, :], nn, ps, st) .- y_test) .^ 2)
	MSE_df[3, :RMSE] = sqrt(MSE_df[3, :MSE])
	MSE_df[4, :State] = "FNN_training_$(size_of_data_split)"
	MSE_df[4, :MSE] = mean((FNN_forward(Xs') .- Matrix(ys')) .^ 2)
	MSE_df[4, :RMSE] = sqrt(MSE_df[4, :MSE])
	MSE_df[4, :Time_s] = dt_fnn
	MSE_df[5, :State] = "FNN_full_training_$(size_of_data_split)"
	MSE_df[5, :MSE] = mean((FNN_forward(X_train') .- Matrix(y_train')) .^ 2)
	MSE_df[5, :RMSE] = sqrt(MSE_df[5, :MSE])
	MSE_df[6, :State] = "FNN_testing_$(size_of_data_split)"
	MSE_df[6, :MSE] = mean((FNN_forward(X_test') .- Matrix(y_test')) .^ 2)
	MSE_df[6, :RMSE] = sqrt(MSE_df[6, :MSE])
	MSE_df[7, :State] = "LC_full_training"
	MSE_df[7, :MSE] = mean((vcat([vcat(log_LC_females...), vcat(log_LC_males...)]...) .- USA_MX_matrix[(USA_MX_matrix[:, 1] .≤ 2000), 4]) .^ 2)
	MSE_df[7, :RMSE] = sqrt(MSE_df[7, :MSE])

	log_LC_test_males_mean, log_LC_test_males_l05, log_LC_test_males_u95, log_LC_test_females_mean, log_LC_test_females_l05, log_LC_test_females_u95, log_LC_test_males_med, log_LC_test_females_med = lee_carter_full_forecast(N)

	MSE_df[8, :State] = "LC_testing"
	MSE_df[8, :MSE] = mean((vcat([vcat(log_LC_test_females_mean...), vcat(log_LC_test_males_mean...)]...) .- USA_MX_matrix[(USA_MX_matrix[:, 1] .> 2000), 4]) .^ 2)
	MSE_df[8, :RMSE] = sqrt(MSE_df[8, :MSE])

	CSV.write("results/BNN_MSE_$(size_of_data_split)_$(N)_psuedo.csv", MSE_df)

	θ_BNN_samples, θ_BNN_for_MAP = sample_BNN_parameters(ch, N, size_of_data_split; pseudo_bnn=Dict("init_params" => init_params, "flag" => true, "random_params" => random_params, "full_params" => Lux.parameterlength(nn)), bkwd)

	CSV.write("results/BNN_full_posterior_samples_$(size_of_data_split)_$(N)_pseudo.csv", DataFrame(Matrix(θ_BNN_samples[:, :, 1]), :auto))

	# Generate and save plots
	for i ∈ 1950:10:2021
		for j ∈ 0:1
			age_plot(i, j, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, θ_BNN_samples, θ_BNN_for_MAP; pseudo="-pseudo")
		end
	end

	pred_interval_score(X_train, y_train, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TRAIN", θ_BNN_samples, θ_BNN_for_MAP)
	pred_interval_score(X_test, y_test, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TEST", θ_BNN_samples, θ_BNN_for_MAP)

	return ch, θ, nn, ps, st, idx
end

# ╔═╡ a28f7cbe-a6ff-4d4b-b327-481c0e81f38d
function get_preds(op, size_of_data_split, N, train_test)
	
	nn = Chain(
		Dense(size(X_train)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	ps, st = Lux.setup(rng, nn)
	
	if train_test == "train"
		X_test_ = deepcopy(X_train)
		samples_one_p = USA_MX_matrix[USA_MX_matrix[:, 1] .≤ 2000, :]
	else
		X_test_ = deepcopy(X_test)
		samples_one_p = USA_MX_matrix[USA_MX_matrix[:, 1] .> 2000, :]
	end

	sample_N = 10_000
	nn_pred_samples = Matrix{Float64}(undef, sample_N, size(X_test_)[1])
	θ_samples = Matrix(DataFrame(CSV.File("results/BNN_full_posterior_samples_$(size_of_data_split)_$(N).csv")))

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_test_', θ_samples[i, :], nn, ps, st)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l01 = quantile.(eachcol(nn_pred_samples), 0.01)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
	nn_pred_u99 = quantile.(eachcol(nn_pred_samples), 0.99)
	nn_pred_sdev = std(nn_pred_samples; dims=1)'

	RESULTS = DataFrame(samples_one_p, [:Year, :Age, :Gender, :Log_Mu])
	if train_test == "train"
		RESULTS.In_sample .= Int.(op)
	end
	RESULTS.BNN_l01 .= nn_pred_l01
	RESULTS.BNN_l05 .= nn_pred_l05
	RESULTS.BNN_mean .= nn_pred_mean
	RESULTS.BNN_median .= nn_pred_median
	RESULTS.BNN_u95 .= nn_pred_u95
	RESULTS.BNN_u99 .= nn_pred_u99
	RESULTS.BNN_sdev .= nn_pred_sdev
	#RESULTS.FNN .= fnn_pred

	CSV.write("results/BNN_results_$(size_of_data_split)_$(N)_$(train_test).csv", RESULTS)
end

# ╔═╡ 9643784e-6dcb-40e5-b3f7-1f75bb43903f
for i ∈ [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
	percent_ = "$(Int(i*100))%"
	one_p = rand(Xoshiro(1456789), Bernoulli(i), size(X_train)[1])
	
	if i < 0.1
		N_length = 5_000
	elseif i < 0.5
		N_length = 7_500
	else
		N_length = 12_500
	end
	
	#get_preds(one_p, percent_, N_length, "train")
	#get_preds(one_p, percent_, N_length, "test")
end

# ╔═╡ Cell order:
# ╠═73791d5e-691d-11ef-1cca-35c52ceada8f
# ╟─c87201c0-fbb6-435b-b956-fd62279f6697
# ╠═a8c535ac-8ccf-4718-8214-c230adcf0b75
# ╠═0c25f42a-513e-4369-96c0-2b2bd0ac6a04
# ╟─07b39487-13d5-4f13-9b86-7f9f8e9b920c
# ╠═f61d2994-d4c7-4cbd-bbb1-09f419c2b62e
# ╟─cb80ed75-b2ce-4fae-8346-93c68c2aeb7f
# ╠═657a1395-6196-420d-86d4-079cc783e827
# ╟─a92d0078-8652-494a-88df-144577f756a5
# ╠═37ed48f0-2be6-47e1-8725-507f1d0b3ee1
# ╠═c33ccbfb-24bf-462f-8d94-67a961decd79
# ╟─d7107f3d-0d5c-476f-b479-fe295828f7b0
# ╠═9a9e8419-c125-433d-8dd9-c2c494af8f3a
# ╟─66063226-138f-4416-9aa5-8e7ad6fa004e
# ╠═540ba303-8602-4d96-97a4-e67dd591f47b
# ╟─4322bdb1-2889-40fe-b9ad-7b4398ebce5a
# ╠═d31f81fd-0908-48d5-aae6-d541f0b0354c
# ╠═b3ec6cfc-95d8-438e-9b8e-3194d00dd475
# ╠═f2061baa-90da-41fa-9c9c-475882c9f475
# ╠═99e91545-ac94-4fe9-8429-4087eeb6d080
# ╠═894c18c2-a145-400e-9b5b-c37af5be0282
# ╠═d9ef687e-914b-4ea7-96b3-7c872a7da035
# ╠═be6f3755-72c9-41d7-aad1-cc53a4addc7e
# ╠═ea534ede-adde-4e69-aa28-45dc7d8b69f8
# ╟─b2ed19f8-9bd8-4965-b518-369c07e3c633
# ╠═45559a00-b2c8-477d-b299-febaa68af5fb
# ╠═48404090-3f4e-4659-9613-79796dc1b977
# ╠═9a2f1a56-8747-4055-aa30-a289a3ee2432
# ╠═de4ec6fd-7d3d-4871-8611-9824c8cabf72
# ╟─a267e96d-d460-4204-a4a9-e4aa7dc29c1e
# ╠═1e172ba4-60a7-4011-b756-8bb47b1d8a92
# ╠═f79c4cff-9664-40e3-8e6f-ca17fdacc532
# ╠═dafa4fa6-8616-4c2e-8f27-c71e29fbe50d
# ╠═35a6b42c-1704-4c0e-b946-75f47b8dd70b
# ╠═000d6486-ba8a-4260-9077-ef9f1d1fe34f
# ╟─6dfb890f-853f-4634-8264-c28ee9d62354
# ╠═0158f329-3f09-4a1a-ab2f-f1595212b236
# ╠═a9fc086b-61d9-4ced-9dca-59a53fa83e39
# ╟─576550d1-fb64-4edc-8610-0e8bd745711d
# ╠═cef77b70-d171-4254-9fe7-0a6b0ce8ede3
# ╠═d1208799-3cf9-4f26-8a4a-b88ae2a03108
# ╠═0e8cc4c5-2ad2-493d-94e1-e312518d6ddd
# ╠═e00a61f5-b488-48c2-a3b5-79f6f1f44081
# ╠═a28f7cbe-a6ff-4d4b-b327-481c0e81f38d
# ╠═9643784e-6dcb-40e5-b3f7-1f75bb43903f
