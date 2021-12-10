# Import usefull packages
using Markdown
using InteractiveUtils
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV, MLJ, MLJLinearModels, OpenML, Statistics

# Import training & test data
begin
	data = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame);
	weather = dropmissing(data)
	test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
end

# Apply multiple logistic regression
# With L2 regularization (Ridge)
begin
	input = select(weather, Not(:precipitation_nextday)); #input
	coerce!(weather, :precipitation_nextday=>Multiclass)

	model = LogisticClassifier(penalty = :l2)
	lambda = [[i/20 for i in 1:20];[j for j in 1:30]]
	selftuning_lambda = TunedModel(model = model,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model, :lambda, values = lambda),
                                   measure = auc)
	selftuning_lambda_mach = fit!(machine(selftuning_lambda, input, weather.precipitation_nextday))
end

begin
	probs_L2 = predict(selftuning_lambda_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
	CSV.write(joinpath(@__DIR__, "..", "results", "logistic_selftuned_L2_regression.csv"), df_L2)
	confusion_matrix(predict_mode(selftuning_lambda_mach, input), weather.precipitation_nextday)
end

