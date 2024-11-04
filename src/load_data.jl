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

  return X_train, y_train, X_train_square_log_males, X_train_square_log_females, X_test, y_test, X_test_square_log_males, X_test_square_log_females, MX_matrix, year_mu, year_sigma
end

# Set data bounds 
train_start_year = 1950
train_end_year = 2000
test_start_year = 2001
test_end_year = 2016
start_age = 0
end_age = 100

#Load the full USA mortality dataset, capping the age range at 100 and year range at 1950.
X_train, y_train, X_train_square_log_males, X_train_square_log_females, X_test, y_test, X_test_square_log_males, X_test_square_log_females, MX_matrix, year_mu, year_sigma = load_data("data/USA_Mx_1x1.csv", 1950, 2000, 2001, 2016, 0, 100);
