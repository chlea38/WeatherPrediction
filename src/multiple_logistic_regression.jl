# Import usefull packages
using Markdown
using InteractiveUtils
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV, MLJ, MLJLinearModels, OpenML, Statistics

# Import training data
begin
	data = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame);
	weather = dropmissing(data)
end

# Apply multiple logistic regression:
begin
	input = select(weather, Not(:precipitation_nextday)); #input
	coerce!(weather, :precipitation_nextday=>Multiclass)
	logistic_mach = fit!(machine(LogisticClassifier(penalty = :none), input, weather.precipitation_nextday))
end

# Compute the error for training data 
"""begin
	probs_train = predict(logistic_mach, input).prob_given_ref.vals
	rmse(probs_train[1], weather.precipitation_nextday)
end"""

# confusion matrix
confusion_matrix(predict_mode(logistic_mach, input), weather.precipitation_nextday)

# import test data
begin
	test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
end

# Predict precipitation next day
begin
	probs = predict(logistic_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs[2])
	CSV.write(joinpath(@__DIR__, "..", "results", "logistic_regression.csv"), df)
end

# With L2 regularization with lambda = 0.6 
begin
	logistic_L2_mach = fit!(machine(LogisticClassifier(penalty = :l2, lambda = 0.6), input, weather.precipitation_nextday))
	probs_L2 = predict(logistic_L2_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
	CSV.write(joinpath(@__DIR__, "..", "results", "logistic_L2_lambda0.6_regression.csv"), df_L2)
	confusion_matrix(predict_mode(logistic_L2_mach, input), weather.precipitation_nextday)
end
