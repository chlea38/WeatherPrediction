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

begin
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
end

# Apply multiple logistic regression
# With L2 regularization (Ridge) 
begin
	model = LogisticClassifier(penalty = :l2)
    lambda = [[i/10 for i in 1:10];[j for j in 2:30];[k*10 for k in 4:10]]
	selftuning_lambda = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :lambda, values = lambda),
                                   measure = auc)
	selftuning_lambda_mach = fit!(machine(selftuning_lambda, input, output))
    # PCA gives far worse results :
    # selftuning_lambda_mach = fit!(machine(@pipeline(PCA(), selftuning_lambda), input,output))
end

# Evaluation with AUC and confusion matrix on full input set
#auc_val = MLJ.auc(predict(selftuning_lambda_mach,input).prob_given_ref.vals[2],output)
confusion_matrix(predict_mode(selftuning_lambda_mach, input), output)


# Save results in CSV file
begin
	probs_L2 = MLJ.predict(selftuning_lambda_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "logistic_selftuned_L2_regression.csv"), df_L2)
end
