# This pipeline performs: 
# - filling missing data with FillImputer
# - train the machine with Logistic Classifier with L2 regularization as penalty (lambda is tuned by optimizing AUC)
# - save prediction in a CSV file

# Import usefull packages
begin
    using Markdown
    using InteractiveUtils
    using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using DataFrames, CSV, MLJ, OpenML, MLJLinearModels, Statistics, MLCourse, LinearAlgebra, Random
end

# Import training & test data and deal with missing data with FillImputer
begin
	weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>OrderedFactor)
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) #Filling imputer data during the fit 
end

# Standardization: ignore ALT_sunshine_4 because creates NaN by dividing by SD=0
# use verbosity to view the means and SDs
begin
    stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), select(weather_filled, Not(:precipitation_nextday))) 
    fit!(stan_mach, verbosity=2) 
    input = select(weather_filled_transf, Not(:precipitation_nextday))
    output = weather_filled_transf.precipitation_nextday
end

# Clean test data
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
    cleaned_test_data_transf = MLJ.transform(stan_mach,cleaned_test_data)
end

# Apply multiple logistic regression with L2 regularization (Ridge) 
begin
	model = LogisticClassifier(penalty = :l1)
	selftuning_lambda = TunedModel(model = model,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model, :lambda, lower=1, upper=10),
                                   measure = auc)
	selftuning_lambda_mach = fit!(machine(selftuning_lambda, input, output))
end

# Evaluation with AUC and confusion matrix on full input set
report(selftuning_lambda_mach).best_model
report(selftuning_lambda_mach).best_history_entry.measurement
confusion_matrix(predict_mode(selftuning_lambda_mach, input), output)

# Save results in CSV file
begin
	probs_L2 = MLJ.predict(selftuning_lambda_mach, cleaned_test_data_transf).prob_given_ref.vals
	N = size(cleaned_test_data_transf)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "logistic_selftuned_L1_reg_stan.csv"), df_L2)
end
