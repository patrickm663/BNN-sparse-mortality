using DataFrames, CSV, HMD, Statistics

function load_data(file_path, train_start_year, train_end_year, test_start_year, test_end_year, start_age, end_age)

  MX_raw = CSV.read(file_path, DataFrame)

  # Get 1950+ and ≤ 100
  MX_raw = MX_raw[(MX_raw.Year .≥ train_start_year) .&& (start_age .≤ MX_raw.Age .≤ end_age), :]

  # For Lee-Carter
  MX_square_log_males = log.(HMD.transform(MX_raw, :Male)[:, 2:end])
  MX_square_log_females = log.(HMD.transform(MX_raw, :Female)[:, 2:end])

  # For NN
  MX_matrix = Matrix{Float64}(undef, (end_age-start_age+1)*(test_end_year-train_start_year+1)*2, 4)

  c = 0
  for i in 0:1
    for j in train_start_year:test_end_year
      for k in start_age:end_age
	c += 1
	MX_matrix[c, 1] = j
	MX_matrix[c, 2] = k
	MX_matrix[c, 3] = i
	# Male = 1
	if i == 1
	  male_MX = log(MX_raw[(MX_raw.Year .== j) .&& (MX_raw.Age .== k), :Male][1])
	  MX_matrix[c, 4] = male_MX
	  # Female = 0
	else
	  female_MX = log(MX_raw[(MX_raw.Year .== j) .&& (MX_raw.Age .== k), :Female][1])
	  MX_matrix[c, 4] = female_MX
	end
      end
    end
  end

  MX_matrix = MX_matrix[MX_matrix[:, 4] .> -Inf, :]

  #Create the Z-transformation scales. We can apply this because we know the minimum and maximum age and years of the observation -- regardless of missing data in between.

  # Training/seen data is 1950-2000
  age_range = start_age:end_age
  year_range = train_start_year:train_end_year

  age_mu, age_sigma = (mean(age_range), std(age_range))
  year_mu, year_sigma = (mean(year_range), std(year_range))

  rescale(x, μ, σ) = (x .* σ) .+ μ

  #Scale the training set and create a 'grid' for the Lee-Carter model to process (on all the data). TODO: explore Lee-Carter with missing values.

  # First get 'full' X_train and y_train for NN
  X_train = MX_matrix[MX_matrix[:, 1] .≤ year_range[end], 1:3]
  X_train[:, 1] = (X_train[:, 1] .- year_mu) ./ year_sigma
  X_train[:, 2] = (X_train[:, 2] .- age_mu) ./ age_sigma
  y_train = MX_matrix[MX_matrix[:, 1] .≤ year_range[end], 4]

  # Get LC train set
  X_train_square_log_males = Matrix(MX_square_log_males[:, year_range .- (year_range[1] - 1)])
  X_train_square_log_females = Matrix(MX_square_log_females[:, year_range .- (year_range[1] - 1)])

  # Test/unseen is everything after 2000 for NN
  X_test = MX_matrix[year_range[end] .< MX_matrix[:, 1], 1:3]
  X_test[:, 1] = (X_test[:, 1] .- year_mu) ./ year_sigma
  X_test[:, 2] = (X_test[:, 2] .- age_mu) ./ age_sigma
  y_test = MX_matrix[year_range[end] .< MX_matrix[:, 1], 4]

  # Get LC test set
  X_test_square_log_males = Matrix(MX_square_log_males[:, (test_start_year:test_end_year) .- (year_range[1] - 1)])
  X_test_square_log_females = Matrix(MX_square_log_females[:, (test_start_year:test_end_year) .- (year_range[1] - 1)])

  return X_train, y_train, X_train_square_log_males, X_train_square_log_females, X_test, y_test, X_test_square_log_males, X_test_square_log_females, MX_matrix, year_mu, year_sigma, age_mu, age_sigma 
end

function load_data_multi_pop(train_start_year, train_end_year, test_start_year, test_end_year, start_age, end_age)

  MX_matrix_complete = Matrix{Float64}(undef, 0, 4)
  ex_mapping = Dict{String, Float64}()
  country_list = HMD.get_countries() |> values |> unique
  for country in country_list
    try
      MX_file_path = "data/$(country)_Mx_1x1.csv"
      EX_file_path = "data/$(country)_Ex_1x1.csv"

      MX_raw = CSV.read(MX_file_path, DataFrame)
      MX_raw = MX_raw[(start_age .≤ MX_raw.Age .≤ end_age) .&& (train_start_year .≤ MX_raw.Year .≤ test_end_year), :]
      EX_raw = CSV.read(EX_file_path, DataFrame)
      EX_raw = EX_raw[(train_start_year .≤ EX_raw.Year .≤ train_end_year), :]

      ex_mapping["$(country)_Males"] = mean(EX_raw.Male) 
      ex_mapping["$(country)_Females"] = mean(EX_raw.Female) 

      local_start_age, local_end_age = extrema(MX_raw.Age)
      local_start_year, local_end_year = extrema(MX_raw.Year)
      MX_matrix = Matrix{Float64}(undef, (local_end_age-local_start_age+1)*(local_end_year-local_start_year+1)*2, 4)

      c = 0
      for i in 0:1
	for j in local_start_year:local_end_year
	  for k in local_start_age:local_end_age
	    c += 1
	    MX_matrix[c, 1] = j
	    MX_matrix[c, 2] = k
	    # Male = 1
	    if i == 1
	      MX_matrix[c, 3] = mean(EX_raw[:, :Male])
	      male_MX = log(MX_raw[(MX_raw.Year .== j) .&& (MX_raw.Age .== k), :Male][1])
	      MX_matrix[c, 4] = male_MX
	      # Female = 0
	    else
	      MX_matrix[c, 3] = mean(EX_raw[:, :Female])
	      female_MX = log(MX_raw[(MX_raw.Year .== j) .&& (MX_raw.Age .== k), :Female][1])
	      MX_matrix[c, 4] = female_MX
	    end
	  end
	end
      end

      MX_matrix = MX_matrix[MX_matrix[:, 4] .> -Inf, :]
      MX_matrix_complete = [MX_matrix_complete; MX_matrix]
    catch
    end
  end

  MX_complete = deepcopy(MX_matrix_complete)
  age_range = start_age:end_age
  year_range = train_start_year:train_end_year

  age_mu, age_sigma = (mean(age_range), std(age_range))
  year_mu, year_sigma = (mean(year_range), std(year_range))
  ex_mu, ex_sigma = (mean(MX_matrix_complete[:, 3]), std(MX_matrix_complete[:, 3]))

  standardise(x, μ, σ) = (x .- μ) ./ σ
  rescale(x, μ, σ) = (x .* σ) .+ μ

  train = MX_complete[MX_complete[:, 1] .< test_start_year, :]
  test = MX_complete[train_end_year .< MX_complete[:, 1], :]
  
  train[:, 1] = standardise(train[:, 1], year_mu, year_sigma)
  train[:, 2] = standardise(train[:, 2], age_mu, age_sigma)
  train[:, 3] = standardise(train[:, 3], ex_mu, ex_sigma)
  test[:, 1] = standardise(test[:, 1], year_mu, year_sigma)
  test[:, 2] = standardise(test[:, 2], age_mu, age_sigma)
  test[:, 3] = standardise(test[:, 3], ex_mu, ex_sigma)

  X_train = train[:, 1:3]
  y_train = train[:, 4]
  X_test = test[:, 1:3]
  y_test = test[:, 4]

  CSV.write("data/Mx_complete_STD.csv", DataFrame(MX_complete, [:Year, :Age, :Ex, :log_Mx]))
  CSV.write("data/Mx_complete.csv", DataFrame(MX_matrix_complete, [:Year, :Age, :Ex, :log_Mx]))
  return X_train, y_train, X_test, y_test, MX_matrix_complete, year_mu, year_sigma, age_mu, age_sigma, ex_mu, ex_sigma, ex_mapping 

end

# Set data bounds 
train_start_year = 1950
train_end_year = 1999
test_start_year = 2000
test_end_year = 2016
start_age = 0
end_age = 99

#Load the full USA mortality dataset, capping the age range at 100 and year range at 1950.
X_train, y_train, X_train_square_log_males, X_train_square_log_females, X_test, y_test, X_test_square_log_males, X_test_square_log_females, MX_matrix, year_mu, year_sigma, age_mu, age_sigma = load_data("data/LUX_Mx_1x1.csv", train_start_year, train_end_year, test_start_year, test_end_year, start_age, end_age);

X_train_c, y_train_c, X_test_c, y_test_c, MX_matrix_c, year_mu_c, year_sigma_c, age_mu_c, age_sigma_c, ex_mu_c, ex_sigma_c, ex_mapping_c = load_data_multi_pop(train_start_year, train_end_year, test_start_year, test_end_year, start_age, end_age);
