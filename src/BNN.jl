include("load_data.jl")

using Lux, Tracker, Optimisers, Functors, DataFrames, Plots, StatsPlots
using Turing, Distributions, ProgressLogging
using Random, LinearAlgebra, ComponentArrays


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


function nn_forward(x, θ, nn, ps, st, σ_MAP; offset=0)
  nn_output = vec(first(nn(x, vector_to_parameters_(θ, ps), st)))
  return nn_output .+ rand(Xoshiro(1456789+offset), Normal(0.0, σ_MAP))
end

function BNN(Xs, ys, N, perc, size_of_data_split; sampling_algorithm="NUTS", save=true)
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

  ## TODO to test mini-batched MCMC idea
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


  # Get MAP
  _, idx = findmax(ch[:lp])
  idx = idx.I[1]

  nn_forward_(x, θ, nn, ps, st) = vec(first(nn(x, vector_to_parameters(θ, ps), st)))

  θ = MCMCChains.group(ch, :parameters).value

  # Save MSE
  MSE_df = DataFrame(:State => ["", "", ""], :MSE => zeros(3), :RMSE => zeros(3), :Time_s => zeros(3))
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


  θ_BNN_samples, θ_BNN_for_MAP = sample_BNN_parameters(ch, N, size_of_data_split)

  #pred_interval_score(X_train, y_train, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TRAIN", θ_BNN_samples, θ_BNN_for_MAP)
  #pred_interval_score(X_test, y_test, ch, nn, ps, st, idx, perc, size_of_data_split, N, tstate, "TEST", θ_BNN_samples, θ_BNN_for_MAP)

  if save == true
    CSV.write("results/full_chains_$(size_of_data_split)_$(N).csv", DataFrame(MCMCChains.group(ch, :parameters).value[:, :, 1], :auto))
    CSV.write("results/summary_chains_$(size_of_data_split)_$(N).csv", DataFrame(describe(ch)[1]))
    savefig(StatsPlots.plot(ch[half_N:end, 1:80:end, :]), "results/chains_plot_$(size_of_data_split)_$(N).png")
    CSV.write("results/BNN_full_posterior_samples_$(size_of_data_split)_$(N).csv", DataFrame(Matrix(θ_BNN_samples[:, :, 1]), :auto))
    CSV.write("results/BNN_MAP_RESULTS_$(size_of_data_split)_$(N).csv", MSE_df)
  end

  return ch, θ, nn, ps, st, idx, θ_BNN_samples, θ_BNN_for_MAP
end

function sample_BNN_parameters(ch, N, size_of_data_split) 
  sample_N = max(N, 10_000)
  posterior_samples = sample(Xoshiro(1456789), ch, sample_N)

  θ_samples = MCMCChains.group(posterior_samples, :parameters).value
  θ_for_MAP = MCMCChains.group(ch, :parameters).value

  return θ_samples, θ_for_MAP
end

function get_preds(size_of_data_split, N, train_test; save=true)

  nn = Chain(
	     Dense(size(X_train)[2] => 8, swish), 
	     Dense(8 => 8, swish),
	     Dense(8 => 8, swish),
	     Dense(8 => 8, swish),
	     Dense(8 => 1))

  ps, st = Lux.setup(Xoshiro(321), nn)

  if train_test == "train"
    X_test_ = deepcopy(X_train)
    samples_one_p = MX_matrix[MX_matrix[:, 1] .≤ 2000, :]
  elseif train_test == "test"
    X_test_ = deepcopy(X_test)
    samples_one_p = MX_matrix[MX_matrix[:, 1] .> 2000, :]
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

  if save == true
    CSV.write("results/BNN_results_$(size_of_data_split)_$(N)_$(train_test).csv", RESULTS)
    CSV.write("results/EXP_BNN_results_$(size_of_data_split)_$(N)_$(train_test).csv", EXP_RESULTS)
  end
end

function age_plot(year, gender, ch, nn, ps, st, idx, perc, size_of_data_split, N, θ_samples, θ_for_MAP)
  if year ≤ 2000
    X_test_ = X_train[(X_train[:, 1] .== (year .- year_mu) ./ year_sigma) .&& (X_train[:, 3] .== gender), :]
    samples_one_p = MX_matrix[MX_matrix[:, 1] .≤ 2000, :][perc, :]
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

  nn_pred_mean = vcat(mean(nn_pred_samples; dims=1)'...)
  nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
  nn_pred_l05 = quantile.(eachcol(nn_pred_samples), 0.05)
  nn_pred_u95 = quantile.(eachcol(nn_pred_samples), 0.95)
  nn_pred_MAP =  nn_forward(X_test_', θ_for_MAP[idx, :], nn, ps, st, σ_MAP)

  p_ = begin
    # Plot front-matter
    if year ≤ 2000
      if gender == 1
	title_ = "Mortality Rates using $(size_of_data_split) of Data \n In-sample: $(year); Males"
      else
	title_ = "Mortality Rates using $(size_of_data_split) of Data \n In-sample: $(year); Females"
      end
    else
      if gender == 1
	title_ = "Mortality Rates using $(size_of_data_split) of Data \n Out-sample: $(year); Males"
      else
	title_ = "Mortality Rates using $(size_of_data_split) of Data \n Out-sample: $(year); Females"
      end
    end
    plot(title=title_, xlab="Age", ylab="log(μ)", label="", legend=:outertopright)

    # Plot BNN mean
    plot!(0:100, nn_pred_mean, label="BNN: Mean", color=:blue, width=2)

    # Plot BNN CI
    plot!(0:100, nn_pred_l05, fillrange=nn_pred_u95, label="BNN: 95% CI", color=:blue, width=0.1, alpha=0.3)

    # Plot BNN median
    plot!(0:100, nn_pred_median, label="BNN: Median", color=:blue, width=2, style=:dashdot)

    # Plot BNN MAP
    plot!(0:100, nn_pred_MAP, label="BNN: MAP", color=:blue, width=2, style=:dot)

    # Plot Observations
    if year ≤ 2000
      scatter!(samples_one_p[:, 2], samples_one_p[:, 4], label="Observed Samples")
    end

    scatter!(0:100, MX_matrix[(MX_matrix[:, 1] .== year) .&& (MX_matrix[:, 3] .== gender), 4], label="Full Underlying Data", color=:black, markershape=:circle, markersize=1.5, ylim=(-10.0, -0.5))

    plot!()
  end

  p_

  savefig(p_, "results/$(year)-$(gender)-$(size_of_data_split)-$(N)-BNN.png")

  return p_
end

begin
  #for i ∈ [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
  for i ∈ [0.005] # Debug
  if i < 0.01
      percent_ = "half-%"
    else
      percent_ = "$(Int(i*100))%"
    end # 1456789
    one_p = rand(Xoshiro(1145689), Bernoulli(i), size(X_train)[1])
    X_train_one_p = X_train[one_p, :]
    y_train_one_p = y_train[one_p]
    samples_one_p_ = MX_matrix[MX_matrix[:, 1] .≤ 2000, :][one_p, :]

    # 0.5% = 2 500, 1% = 5 000, 5% = 7 500, 10% = 10 000, 25% = 15 000
    i = 0
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
    else # Debugging
      N_length = 100
    end


    ch_p, θ_p, nn_p, ps_p, st_p, idx_p, θ_BNN_samples_p, θ_BNN_for_MAP_p = BNN(X_train_one_p, y_train_one_p, N_length, one_p, percent_; save=true)
    _ = get_preds(percent_, N_length, "train"; save=true)
    _ = get_preds(percent_, N_length, "test"; save=true)

    for i ∈ 1950:10:2016
      for j ∈ 0:1
        age_plot(i, j, ch_p, nn_p, ps_p, st_p, idx_p, one_p, percent_, N_length, θ_BNN_samples_p, θ_BNN_for_MAP_p)
      end
    end
  end

end