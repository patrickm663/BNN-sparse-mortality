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
	using Lux, Tracker, Enzyme, Optimisers, Functors, Zygote
	using DataFrames, CSV, HMD, Plots, StatsPlots
	using Turing, Distributions, ProgressLogging
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
function nn_forward(x, θ, nn, ps, st, σ_MAP; offset=0)
	nn_output = vec(first(nn(x, vector_to_parameters_(θ, ps), st)))
	return nn_output .+ rand(Xoshiro(1456789+offset), Normal(0.0, σ_MAP))
end

# ╔═╡ 6dfb890f-853f-4634-8264-c28ee9d62354
md"""
Loops over different percentages (1%, 5%, etc.) of the training data and call the "BNN" function.
"""

# ╔═╡ 0158f329-3f09-4a1a-ab2f-f1595212b236
begin
	for i ∈ [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
		if i < 0.01
			percent_ = "half-%"
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

		#ch_one_p, θ_one_p, nn_one_p, ps_one_p, st_one_p, idx_one_p = BNN(X_train_one_p, y_train_one_p, N_length, one_p, percent_)
		#ch_one_p, θ_one_p, nn_one_p, ps_one_p, st_one_p, idx_one_p = BNN_pseudo(X_train_one_p, y_train_one_p, N_length, one_p, percent_, 5; bkwd=false)

	end
end

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
	σ_MAP = MCMCChains.group(ch, :σ).value[idx, 1]

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_', θ_samples[i, :], nn, ps, st, σ_MAP; offset=i)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
	nn_pred_MAP =  nn_forward(X_', θ_for_MAP[idx, :], nn, ps, st, σ_MAP)

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
function sample_BNN_parameters(ch, N, size_of_data_split; pseudo_bnn=Dict("init_params" => 0, "flag" => false, "random_params" => 0, "full_params" => 0), bkwd=false)
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
	σ_MAP = MCMCChains.group(ch, :σ).value[idx, 1]

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_test_', θ_samples[i, :], nn, ps, st, σ_MAP; offset=i)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
	nn_pred_MAP =  nn_forward(X_test_', θ_for_MAP[idx, :], nn, ps, st, σ_MAP)

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
	ps, st = Lux.setup(Xoshiro(1456789), nn)

	fnn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	ps_fnn, st_fnn = Lux.setup(Xoshiro(1456789), fnn)

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
	tstate, losses_ = train_FNN(tstate, vjp_rule, (Xs', Matrix(ys')), max(3N, 8_000))
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
			NUTS(0.9; adtype=AutoTracker()),
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
	MSE_df[1, :MSE] = mean((nn_forward_(Xs', θ[idx, :], nn, ps, st) .- ys) .^ 2)
	MSE_df[1, :RMSE] = sqrt(MSE_df[1, :MSE])
	MSE_df[1, :Time_s] = dt_bnn
	MSE_df[2, :State] = "BNN_full_training_$(size_of_data_split)"
	MSE_df[2, :MSE] = mean((nn_forward_(X_train', θ[idx, :], nn, ps, st) .- y_train) .^ 2)
	MSE_df[2, :RMSE] = sqrt(MSE_df[2, :MSE])
	MSE_df[3, :State] = "BNN_testing_$(size_of_data_split)"
	MSE_df[3, :MSE] = mean((nn_forward_(X_test', θ[idx, :], nn, ps, st) .- y_test) .^ 2)
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

	θ_BNN_samples, θ_BNN_for_MAP = sample_BNN_parameters(ch, N, size_of_data_split; pseudo_bnn=Dict("init_params" => 0, "flag" => false, "random_params" => 0), bkwd=false)

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
	ps, st = Lux.setup(Xoshiro(1456789), nn)

	fnn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	ps_fnn, st_fnn = Lux.setup(Xoshiro(1456789), fnn)

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
	MSE_df[1, :MSE] = mean((nn_forward_(Xs', θ[idx, :], nn, ps, st) .- ys) .^ 2)
	MSE_df[1, :RMSE] = sqrt(MSE_df[1, :MSE])
	MSE_df[1, :Time_s] = dt_bnn
	MSE_df[2, :State] = "BNN_full_training_$(size_of_data_split)"
	MSE_df[2, :MSE] = mean((nn_forward_(X_train', θ[idx, :], nn, ps, st) .- y_train) .^ 2)
	MSE_df[2, :RMSE] = sqrt(MSE_df[2, :MSE])
	MSE_df[3, :State] = "BNN_testing_$(size_of_data_split)"
	MSE_df[3, :MSE] = mean((nn_forward_(X_test', θ[idx, :], nn, ps, st) .- y_test) .^ 2)
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

# ╔═╡ 0775b300-6cff-4dd8-bb87-3f79f2cfef47
function fnn_prediction_interval(Xs, ys, percent_s; B=100, b=0.75, save=true)
	# Model structure
	fnn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1))

	dev_cpu = cpu_device()
	dev_gpu = cpu_device() # CPU throughout

	ps_fnn, st_fnn = Lux.setup(Xoshiro(1456789), fnn) |> dev_gpu

	opt = Adam(0.001f0)
	loss_function = MSELoss()
	tstate = Training.TrainState(fnn, ps_fnn, st_fnn, opt)
	vjp_rule = AutoZygote() #AutoEnzyme() #AutoTracker()

	function train_FNN(tstate::Training.TrainState, vjp, data, epochs; loss_func=loss_function)
		losses = []
		data = data .|> gpu_device()
		# Check for early-stopping
		@inbounds for epoch ∈ 1:epochs
			_, loss, _, tstate = Training.single_train_step!(vjp, loss_func, data, tstate)
			push!(losses, loss)
			if (epoch ≥ 250) && (epoch % 10 == 1) && (losses[epoch] ≥ losses[(epoch-10)])
				return tstate
			end
		end
	    return tstate
	end

	FNN_forward(X_, ts_; pars=ts_.parameters) = dev_cpu(Lux.apply(ts_.model, dev_gpu(X_), pars, ts_.states)[1])

	function parameters_to_vectors(nn_param)
		init_params = zeros(Lux.parameterlength(nn_param))
		m = 0
		@inbounds for i ∈ 1:length(nn_param)
			@inbounds for j in 1:2 # either weights or bias
				init_params_ = vcat(deepcopy(nn_param[i][j])...)
				@inbounds for l ∈ 1:length(init_params_)
					m = m + 1
					init_params[m] = init_params_[l]
				end
			end
		end
		return init_params
	end

	bootstrap_epochs = 800#max(800, Int(floor(size(Xs)[1]*1.2)))
	
	bootstrap_samples_train_s = Matrix{Float64}(undef, B, size(Xs)[1])
	bootstrap_samples_train = Matrix{Float64}(undef, B, size(X_train)[1])
	bootstrap_samples_test = Matrix{Float64}(undef, B, size(X_test)[1])
	
	bootstrap_params = Matrix{Float64}(undef, B, Lux.parameterlength(tstate.parameters))
	
	b_size = Int(floor(size(Xs)[1] * b))
	
	t_fnn = time()
	@progress for i ∈ 1:B
		rng_Xoshiro = Xoshiro(i+100)
		bootstrap_p = rand(rng_Xoshiro, axes(Xs, 1), b_size)
		bootstrap_Xs = Xs[bootstrap_p, :]
		bootstrap_ys = ys[bootstrap_p]
		ps_fnn, st_fnn = Lux.setup(Xoshiro(1456789-2i), fnn) |> dev_gpu
		tstate = Training.TrainState(fnn, ps_fnn, st_fnn, opt)
		tstate_bootstrap = train_FNN(tstate, vjp_rule, (bootstrap_Xs', Matrix(bootstrap_ys')), bootstrap_epochs)
		bootstrap_samples_train_s[i, :] .= FNN_forward(Xs', tstate_bootstrap)'
		bootstrap_samples_train[i, :] .= FNN_forward(X_train', tstate_bootstrap)'
		bootstrap_samples_test[i, :] .= FNN_forward(X_test', tstate_bootstrap)'
		bootstrap_params[i, :] .= parameters_to_vectors(dev_cpu(tstate_bootstrap.parameters))
	end

	#bootstrap_params_sample = bootstrap_params[rand(Xoshiro(1456789), axes(bootstrap_params, 1), posterior_samples), :]

	#@inbounds for i ∈ 1:B
	#	bootstrap_pars = vector_to_parameters_(vec(bootstrap_params[i, :]'), dev_cpu(ps_fnn))
	#	bootstrap_samples_train_s[i, :] .= FNN_forward(Xs', tstate; pars=bootstrap_pars)'
	#	bootstrap_samples_train[i, :] .= FNN_forward(X_train', tstate; pars=bootstrap_pars)'
	#	bootstrap_samples_test[i, :] .= FNN_forward(X_test', tstate; pars=bootstrap_pars)'
	#end
	
	bootstrap_mean_train_s = mean(bootstrap_samples_train_s; dims=1)'
	bootstrap_var_train_s = var(bootstrap_samples_train_s; dims=1)'
	bootstrap_mean_train = mean(bootstrap_samples_train; dims=1)'
	bootstrap_var_train = var(bootstrap_samples_train; dims=1)'

	bootstrap_mean_test = mean(bootstrap_samples_test; dims=1)'
	bootstrap_var_test = var(bootstrap_samples_test; dims=1)'

	residuals_train_s = max.((ys .- bootstrap_mean_train_s) .^ 2 .- bootstrap_var_train_s, 0)

	fnn_resid = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 1, exp))

	ps_fnn_resid, st_fnn_resid = Lux.setup(Xoshiro(1456789), fnn_resid) |> dev_gpu

	tstate_resid_ = Training.TrainState(fnn_resid, ps_fnn_resid, st_fnn_resid, opt)

	tstate_resid = train_FNN(tstate_resid_, vjp_rule, (Xs', Matrix(residuals_train_s')), bootstrap_epochs)

	dt_fnn = time() - t_fnn

	residual_pred_train = FNN_forward(X_train', tstate_resid)'
	total_var_train = residual_pred_train .+ bootstrap_var_train
	y_95_train = bootstrap_mean_train .+ (1.96*sqrt.(total_var_train))
	y_05_train = bootstrap_mean_train .- (1.96*sqrt.(total_var_train))

	residual_pred_test = FNN_forward(X_test', tstate_resid)'
	total_var_test = residual_pred_test .+ bootstrap_var_test
	y_95_test = bootstrap_mean_test .+ (1.96*sqrt.(total_var_test))
	y_05_test = bootstrap_mean_test .- (1.96*sqrt.(total_var_test))

	RESULTS_TRAIN = DataFrame(USA_MX_matrix[USA_MX_matrix[:, 1] .≤ 2000, :], [:Year, :Age, :Gender, :Log_Mu])
	RESULTS_TEST = DataFrame(USA_MX_matrix[USA_MX_matrix[:, 1] .> 2000, :], [:Year, :Age, :Gender, :Log_Mu])

	RESULTS_TRAIN.lb .= y_05_train
	RESULTS_TRAIN.mean .= bootstrap_mean_train
	RESULTS_TRAIN.ub .= y_95_train

	RESULTS_TEST.lb .= y_05_test
	RESULTS_TEST.mean .= bootstrap_mean_test
	RESULTS_TEST.ub .= y_95_test

	EXP_RESULTS = DataFrame(
		:log_mse_train => 0.0,
		:mse_train => 0.0,
		:log_mae_train => 0.0,
		:mae_train => 0.0,
		:picp_train => 0.0,
		:log_mwip_train => 0.0,
		:mwip_train => 0.0,
		:log_mse_test => 0.0,
		:mse_test => 0.0,
		:log_mae_test => 0.0,
		:mae_test => 0.0,
		:picp_test => 0.0,
		:log_mwip_test => 0.0,
		:mwip_test => 0.0,
		:train_time => 0.0
	)

	EXP_RESULTS.log_mse_train .= mean((RESULTS_TRAIN.mean .- RESULTS_TRAIN.Log_Mu) .^ 2)
	EXP_RESULTS.mse_train .= mean((exp.(RESULTS_TRAIN.mean) .- exp.(RESULTS_TRAIN.Log_Mu)) .^ 2)
	EXP_RESULTS.log_mae_train .= mean(abs.(RESULTS_TRAIN.mean .- RESULTS_TRAIN.Log_Mu))
	EXP_RESULTS.mae_train .= mean(abs.(exp.(RESULTS_TRAIN.mean) .- exp.(RESULTS_TRAIN.Log_Mu)))
	EXP_RESULTS.picp_train .= mean(RESULTS_TRAIN.ub .≥ RESULTS_TRAIN.Log_Mu .≥ RESULTS_TRAIN.lb)
	EXP_RESULTS.log_mwip_train .= mean(RESULTS_TRAIN.ub .- RESULTS_TRAIN.lb)
	EXP_RESULTS.mwip_train .= mean(exp.(RESULTS_TRAIN.ub) .- exp.(RESULTS_TRAIN.lb))
	
	EXP_RESULTS.log_mse_test .= mean((RESULTS_TEST.mean .- RESULTS_TEST.Log_Mu) .^ 2)
	EXP_RESULTS.mse_test .= mean((exp.(RESULTS_TEST.mean) .- exp.(RESULTS_TEST.Log_Mu)) .^ 2)
	EXP_RESULTS.log_mae_test .= mean(abs.(RESULTS_TEST.mean .- RESULTS_TEST.Log_Mu))
	EXP_RESULTS.mae_test .= mean(abs.(exp.(RESULTS_TEST.mean) .- exp.(RESULTS_TEST.Log_Mu)))
	EXP_RESULTS.picp_test .= mean(RESULTS_TEST.ub .≥ RESULTS_TEST.Log_Mu .≥ RESULTS_TEST.lb)
	EXP_RESULTS.log_mwip_test .= mean(RESULTS_TEST.ub .- RESULTS_TEST.lb)
	EXP_RESULTS.mwip_test .= mean(exp.(RESULTS_TEST.ub) .- exp.(RESULTS_TEST.lb))

	EXP_RESULTS.train_time .= dt_fnn

	if save == true
		CSV.write("results/FNN_bootrap_params_$(percent_s)_$(B).csv", DataFrame(bootstrap_params, :auto))
		CSV.write("results/FNN_PI_TRAIN_$(percent_s)_$(B).csv", RESULTS_TRAIN)
		CSV.write("results/FNN_PI_TEST_$(percent_s)_$(B).csv", RESULTS_TEST)
		CSV.write("results/FNN_PI_RESULTS_$(percent_s)_$(B).csv", EXP_RESULTS)
	end
	
	return bootstrap_samples_train, bootstrap_samples_test
end

# ╔═╡ a28f7cbe-a6ff-4d4b-b327-481c0e81f38d
function get_preds(size_of_data_split, N, train_test)
	
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
	elseif train_test == "test"
		X_test_ = deepcopy(X_test)
		samples_one_p = USA_MX_matrix[USA_MX_matrix[:, 1] .> 2000, :]
	end

	sample_N = 10_000
	nn_pred_samples = Matrix{Float64}(undef, sample_N, size(X_test_)[1])
	θ_samples = Matrix(DataFrame(CSV.File("results/BNN_full_posterior_samples_$(size_of_data_split)_$(N).csv")))
	σ_MAP = 0.04

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = nn_forward(X_test_', θ_samples[i, :], nn, ps, st, σ_MAP; offset=i)
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l01 = quantile.(eachcol(nn_pred_samples), 0.01)
	nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.025)
	nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.975)
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

	EXP_RESULTS = DataFrame(
		:MSE_mean => 0.0, 
		:MSE_median => 0.0, 
		:MAE_mean => 0.0, 
		:MAE_median => 0.0, 
		:PICP => 0.0, 
		:MIPW => 0.0,
		:MSE_korn_mean => 0.0, 
		:MSE_korn_median => 0.0, 
		:MAE_korn_mean => 0.0, 
		:MAE_korn_median => 0.0, 
		:PICP_korn => 0.0, 
		:MIPW_korn  => 0.0,
		:MSE_korn_full_mean => 0.0, 
		:MSE_korn_full_median => 0.0, 
		:MAE_korn_full_mean => 0.0, 
		:MAE_korn_full_median => 0.0, 
		:PICP_korn_full => 0.0, 
		:MIPW_korn_full  => 0.0,
		:MSE_perla_mean => 0.0, 
		:MSE_perla_median => 0.0, 
		:MAE_perla_mean => 0.0, 
		:MAE_perla_median => 0.0, 
		:PICP_perla => 0.0, 
		:MIPW_perla  => 0.0,
		:MSE_nigri_mean => 0.0, 
		:MSE_nigri_median => 0.0, 
		:MAE_nigri_mean => 0.0, 
		:MAE_nigri_median => 0.0, 
		:PICP_nigri => 0.0, 
		:MIPW_nigri  => 0.0,
	)
	if train_test == "test"
		RESULTS_korn = RESULTS[(60 .≤ RESULTS.Age .≤ 89) .&& (2007 .≤ RESULTS.Year .≤ 2016), :]
		RESULTS_korn_full = RESULTS[(2 .≤ RESULTS.Age .≤ 98) .&& (2007 .≤ RESULTS.Year .≤ 2016), :]
	else
		RESULTS_korn = RESULTS[(60 .≤ RESULTS.Age .≤ 89) .&& (RESULTS.Year .≤ 2006), :]
		RESULTS_korn_full = RESULTS[(2 .≤ RESULTS.Age .≤ 98) .&& (RESULTS.Year .≤ 2006), :]
	end
	
	RESULTS_perla = RESULTS[(RESULTS.Age .≤ 99) .&& (RESULTS.Year .≤ 2016), :]
	RESULTS_nigri = RESULTS[(RESULTS.Year .≤ 2015), :]


	EXP_RESULTS.MSE_mean .= mean((exp.(RESULTS.Log_Mu) .- exp.(RESULTS.BNN_mean)) .^ 2)
	EXP_RESULTS.MSE_median .= mean((exp.(RESULTS.Log_Mu) .- exp.(RESULTS.BNN_median)) .^ 2)
	EXP_RESULTS.MAE_mean .= mean(abs.(exp.(RESULTS.Log_Mu) .- exp.(RESULTS.BNN_mean)))
	EXP_RESULTS.MAE_median .= mean(abs.(exp.(RESULTS.Log_Mu) .- exp.(RESULTS.BNN_median)))
	EXP_RESULTS.MIPW .= mean(exp.(RESULTS.BNN_u95) .- exp.(RESULTS.BNN_l05))
	EXP_RESULTS.PICP .= mean(exp.(RESULTS.BNN_u95) .≥ exp.(RESULTS.Log_Mu) .≥ exp.(RESULTS.BNN_l05))
	# Korn et al.
	EXP_RESULTS.MSE_korn_mean .= mean((exp.(RESULTS_korn.Log_Mu) .- exp.(RESULTS_korn.BNN_mean)) .^ 2)
	EXP_RESULTS.MSE_korn_median .= mean((exp.(RESULTS_korn.Log_Mu) .- exp.(RESULTS_korn.BNN_median)) .^ 2)
	EXP_RESULTS.MAE_korn_mean .= mean(abs.(exp.(RESULTS_korn.Log_Mu) .- exp.(RESULTS_korn.BNN_mean)))
	EXP_RESULTS.MAE_korn_median .= mean(abs.(exp.(RESULTS_korn.Log_Mu) .- exp.(RESULTS_korn.BNN_median)))
	EXP_RESULTS.MIPW_korn .= mean(exp.(RESULTS_korn.BNN_u95) .- exp.(RESULTS_korn.BNN_l05))
	EXP_RESULTS.PICP_korn .= mean(exp.(RESULTS_korn.BNN_u95) .≥ exp.(RESULTS_korn.Log_Mu) .≥ exp.(RESULTS_korn.BNN_l05))
	# Korn et al. full age range
	EXP_RESULTS.MSE_korn_full_mean .= mean((exp.(RESULTS_korn_full.Log_Mu) .- exp.(RESULTS_korn_full.BNN_mean)) .^ 2)
	EXP_RESULTS.MSE_korn_full_median .= mean((exp.(RESULTS_korn_full.Log_Mu) .- exp.(RESULTS_korn_full.BNN_median)) .^ 2)
	EXP_RESULTS.MAE_korn_full_mean .= mean(abs.(exp.(RESULTS_korn_full.Log_Mu) .- exp.(RESULTS_korn_full.BNN_mean)))
	EXP_RESULTS.MAE_korn_full_median .= mean(abs.(exp.(RESULTS_korn_full.Log_Mu) .- exp.(RESULTS_korn_full.BNN_median)))
	EXP_RESULTS.MIPW_korn_full .= mean(exp.(RESULTS_korn_full.BNN_u95) .- exp.(RESULTS_korn_full.BNN_l05))
	EXP_RESULTS.PICP_korn_full .= mean(exp.(RESULTS_korn_full.BNN_u95) .≥ exp.(RESULTS_korn_full.Log_Mu) .≥ exp.(RESULTS_korn_full.BNN_l05))
	# Perla et al.
	EXP_RESULTS.MSE_perla_mean .= mean((exp.(RESULTS_perla.Log_Mu) .- exp.(RESULTS_perla.BNN_mean)) .^ 2)
	EXP_RESULTS.MSE_perla_median .= mean((exp.(RESULTS_perla.Log_Mu) .- exp.(RESULTS_perla.BNN_median)) .^ 2)
	EXP_RESULTS.MAE_perla_mean .= mean(abs.(exp.(RESULTS_perla.Log_Mu) .- exp.(RESULTS_perla.BNN_mean)))
	EXP_RESULTS.MAE_perla_median .= mean(abs.(exp.(RESULTS_perla.Log_Mu) .- exp.(RESULTS_perla.BNN_median)))
	EXP_RESULTS.MIPW_perla .= mean(exp.(RESULTS_perla.BNN_u95) .- exp.(RESULTS_perla.BNN_l05))
	EXP_RESULTS.PICP_perla .= mean(exp.(RESULTS_perla.BNN_u95) .≥ exp.(RESULTS_perla.Log_Mu) .≥ exp.(RESULTS_perla.BNN_l05))
	# Nigri et al.
	EXP_RESULTS.MSE_nigri_mean .= mean(((RESULTS_nigri.Log_Mu) .- (RESULTS_nigri.BNN_mean)) .^ 2)
	EXP_RESULTS.MSE_nigri_median .= mean(((RESULTS_nigri.Log_Mu) .- (RESULTS_nigri.BNN_median)) .^ 2)
	EXP_RESULTS.MAE_nigri_mean .= mean(abs.((RESULTS_nigri.Log_Mu) .- (RESULTS_nigri.BNN_mean)))
	EXP_RESULTS.MAE_nigri_median .= mean(abs.((RESULTS_nigri.Log_Mu) .- (RESULTS_nigri.BNN_median)))
	EXP_RESULTS.MIPW_nigri .= mean((RESULTS_nigri.BNN_u95) .- (RESULTS_nigri.BNN_l05))
	EXP_RESULTS.PICP_nigri .= mean((RESULTS_nigri.BNN_u95) .≥ (RESULTS_nigri.Log_Mu) .≥ (RESULTS_nigri.BNN_l05))

	CSV.write("results/BNN_results_$(size_of_data_split)_$(N)_$(train_test).csv", RESULTS)
	CSV.write("results/EXP_BNN_results_$(size_of_data_split)_$(N)_$(train_test).csv", EXP_RESULTS)
end

# ╔═╡ 9643784e-6dcb-40e5-b3f7-1f75bb43903f
begin
	for i ∈ [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
		if i < 0.01
			percent_ = "half-p"
		else
			percent_ = "$(Int(i*100))"
		end
		bp = rand(Xoshiro(1456789), Bernoulli(i), size(X_train)[1])
		X_train_b = X_train[bp, :]
		y_train_b = y_train[bp]
		
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
		else
			N_length = 100 # Debugging
		end
		fnn_prediction_interval(X_train_b, y_train_b, percent_; B=N_length)
		#get_preds(one_p, percent_, N_length, "train")
		#get_preds(percent_, N_length, "test")
	end
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
# ╠═0775b300-6cff-4dd8-bb87-3f79f2cfef47
# ╠═a28f7cbe-a6ff-4d4b-b327-481c0e81f38d
# ╠═9643784e-6dcb-40e5-b3f7-1f75bb43903f
