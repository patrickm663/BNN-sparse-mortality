include("load_data.jl")

using Lux, Optimisers, Zygote, Enzyme, ComponentArrays, LuxCUDA
using Random, LinearAlgebra, Distributions
using ProgressLogging, Plots

function train_single_FNN(Xs, ys, epochs, batch)

  p_ = rand(Xoshiro(1456789), Bernoulli(1.0), size(Xs)[1])
  Xs = Xs[p_, :]
  ys = ys[p_]
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
  tstate_ = Training.TrainState(fnn, ps_fnn, st_fnn, opt)
  vjp_rule = AutoZygote() # ReverseDiff backend

  function train_FNN(tstate::Training.TrainState, vjp, data, epochs; loss_func=loss_function)
    ls = []
    data = data .|> dev_gpu
    # Check for early-stopping
    for epoch ∈ 1:epochs
      _, loss, _, tstate = Training.single_train_step!(vjp, loss_func, data, tstate)
      push!(ls, loss)
      #if (epoch ≥ 250) && (epoch % 10 == 1) && (losses[epoch] ≥ losses[(epoch-10)])
	#return tstate
      #end
    end
    return tstate, ls
  end

  FNN_forward(X_, ts_; pars=ts_.parameters) = dev_cpu(Lux.apply(ts_.model, dev_gpu(X_ |> f32), pars, ts_.states)[1])

  losses = []
  @time for i in 1:batch:length(ys)
    min_len = min(i+batch, length(ys))
    tstate_, loss_ = train_FNN(tstate_, vjp_rule, (Xs[i:min_len, :]' |> f32, Matrix(ys[i:min_len]' |> f32)), epochs)
    losses = vcat(losses, loss_)
  end

  y_train_pred = FNN_forward(X_train_c', tstate_)'
  y_test_pred = FNN_forward(X_test_c', tstate_)'

  EXP_MSE(y, y_) = mean((exp.(y) .- exp.(y_)) .^ 2)

  RESULTS = DataFrame("MSE_TRAIN" => [0.0], "MSE_TEST" => [0.0])

  RESULTS[1, :MSE_TRAIN] = EXP_MSE(y_train_c, y_train_pred)
  RESULTS[1, :MSE_TEST] = EXP_MSE(y_test_c, y_test_pred)

  CSV.write("results/FNN_multi_pop_$(epochs)_$(batch).csv", RESULTS)
  half_losses = Int(floor(length(losses)/2))
  savefig(plot(half_losses:length(losses), losses[half_losses:end], ylab="MSE", xlab="Epochs"), "results/FNN_multi_pop_losses_$(epochs)_$(batch).png")

  function age_plot(year, country, X)
    X_feat = X[(X[:, 1] .== (year-year_mu_c)/year_sigma_c) .&& X[:, 3] .== (ex_country_c[country]-ex_mu_c)/ex_sigma_c, :]
    y_actuals = MX_matrix_c[(MX_matrix_c[:, 1] .== year) .&& (MX_matrix_c[:, 3] .== ex_country_c[country]), 4]

    y_pred = FNN_forward(X_feat', tstate_)'

    plt = begin
      plot(xlab="Age", ylab="log mu", title="$(country) - $(year)")
      scatter!(start_age:end_age, y_actuals, label="Actuals")
      plot!(start_age_end_age, y_pred, label="Predictions")
    end

    savefig(plt, "results/FNN_multi_pop_model$(country)_$(year)_$(epochs)_$(batch).png")

  end

  for cntry in keys(ex_mapping_c)[1:2]
    for i in train_year_start:5:test_year_end
      if i > train_year_end
	age_plot(i, cntry, X_test_c) 
      else
	age_plot(i, cntry, X_train_c) 
      end
    end

  end

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
    vjp_rule = AutoZygote() # ReverseDiff backend

    function train_FNN(tstate::Training.TrainState, vjp, data, epochs; loss_func=loss_function)
      losses = []
      data = data .|> dev_gpu
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

    bootstrap_epochs = 5_000#max(800, Int(floor(size(Xs)[1]*1.2)))

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

    bootstrap_mean_train_s = mean(bootstrap_samples_train_s; dims=1)'
    bootstrap_var_train_s = var(bootstrap_samples_train_s; dims=1)'
    bootstrap_mean_train = mean(bootstrap_samples_train; dims=1)'
    bootstrap_var_train = var(bootstrap_samples_train; dims=1)'

    bootstrap_mean_test = mean(bootstrap_samples_test; dims=1)'
    bootstrap_var_test = var(bootstrap_samples_test; dims=1)'

    # Set-up and train the residuals NN
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

    RESULTS_TRAIN = DataFrame(MX_matrix[MX_matrix[:, 1] .≤ 2000, :], [:Year, :Age, :Gender, :Log_Mu])
    RESULTS_TEST = DataFrame(MX_matrix[MX_matrix[:, 1] .> 2000, :], [:Year, :Age, :Gender, :Log_Mu])

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


  for i ∈ [1.0]#[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
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
      N_length = 500#15_000
    else
      N_length = 1_000 # Debugging
    end
    #fnn_prediction_interval(X_train_b, y_train_b, percent_; B=N_length)
    train_single_FNN(X_train_c, y_train_c, 2^9, 2^15)
  end
