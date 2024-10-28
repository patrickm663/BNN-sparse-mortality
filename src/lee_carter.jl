include("load_data.jl")
using LinearAlgebra, Random, Distributions

#Create a function to perform SVD to construct the Lee-Carter model.
function lee_carter(d)
  log_A = Matrix(d)
  αₓ = mean(log_A; dims=2)
  log_A_std = log_A .- αₓ
  U, λ, Vt = svd(log_A_std; full=true)
  bₓ = U[:, 1]
  kₜ = (λ[1] .* Vt[:, 1])
  
  # Apply scaling per Richman et al. (2019)
  b_sum = sum(bₓ) 
  k_sum = sum(kₜ)
  αₓ = αₓ .+ k_sum .* bₓ
  bₓ = bₓ / b_sum
  kₜ = (kₜ .- k_sum) .* b_sum

  log_LC = αₓ .+ bₓ * kₜ'

  return log_LC, αₓ, bₓ, kₜ
end

function lee_carter_sigma_mle(kappa, t)
  μ = (kappa[end] - kappa[1]) ./ (length(kappa)-1)
  σ = 0.0
  for i in 1:(t-1)
    σ += (kappa[t+1] - kappa[t] - μ)^2
  end
  return μ, σ/(t-1)
end

#Create a function to forecast Kappa t years into the future, given the existing values of Kappa.
function lee_carter_forecast(kappa, t, N)
  # Return a matrix with N random samples of length t

  kappa_proj = Matrix{Float64}(undef, t, N)
  μ, σ = lee_carter_sigma_mle(kappa, t)

  ϵ = 0

  @inbounds for s in 1:N
    ϵ = ϵ + 1
    @inbounds for pr in 1:t
      if pr == 1
	kappa_proj[pr, s] = kappa[end]
      else
	kappa_proj[pr, s] = kappa_proj[pr-1, s] + rand(Xoshiro(1456789+ϵ), Normal(μ, sqrt(σ)))
      end
    end
  end

  return kappa_proj
end

function lee_carter_full_forecast(sample_N, test_year_start, test_year_end)

  pred_length = test_year_end - test_year_end + 1

  lc_pred_males_samples = lee_carter_forecast(kappa_males, pred_length, sample_N)

  kappa_males_forecast_mean = mean(lc_pred_males_samples; dims=2)
  log_LC_test_males_mean = vcat((alpha_males .+ beta_males .* kappa_males_forecast_mean')...)

  kappa_males_forecast_l05 = quantile.(eachrow(lc_pred_males_samples), 0.95)
  log_LC_test_males_l05 = vcat((alpha_males .+ beta_males .* kappa_males_forecast_l05')...)

  kappa_males_forecast_med = quantile.(eachrow(lc_pred_males_samples), 0.5)
  log_LC_test_males_med = vcat((alpha_males .+ beta_males .* kappa_males_forecast_med')...)

  kappa_males_forecast_u95 = quantile.(eachrow(lc_pred_males_samples), 0.05)
  log_LC_test_males_u95 = vcat((alpha_males .+ beta_males .* kappa_males_forecast_u95')...)

  lc_pred_females_samples = lee_carter_forecast(kappa_females, pred_length, sample_N)

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

#Create a Male and Female version. This is because Lee-Carter does not have a way to handle features other than age and year.
log_LC_males, alpha_males, beta_males, kappa_males = lee_carter(X_train_square_log_males)
log_LC_females, alpha_females, beta_females, kappa_females = lee_carter(X_train_square_log_females)

log_LC_test_males_mean, log_LC_test_males_l05, log_LC_test_males_u95, log_LC_test_females_mean, log_LC_test_females_l05, log_LC_test_females_u95, log_LC_test_males_med, log_LC_test_females_med = lee_carter_full_forecast(1, test_start_year, test_end_year)

