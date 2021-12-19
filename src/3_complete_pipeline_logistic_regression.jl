# This complete pipeline perform: 
# - filling missing data with FillImputer
# - train the machine with PCA() and Logistic Classifier
# with L2 regularization as penalty (lambda is tuned by optimizing AUC)
# - save prediction in a CSV file

# Import usefull packages
begin
    using Markdown
    using InteractiveUtils
    using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    #import GLMNet: glmnet
    #using StatsPlots
    #using MLJLinearModels
    using DataFrames, CSV, MLJ, OpenML, MLJLinearModels, Statistics, MLCourse, LinearAlgebra, Random
end

# Import training & test data and deal with missing data with FillImputer
begin
	weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    #coerce!(weather, :precipitation_nextday=>Multiclass)
    coerce!(weather, :precipitation_nextday=>OrderedFactor)
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) #Filling imputer data during the fit 
end

"""# Standardization
# contient des Infs or NaNs
# verbosity pour afficher les moyennes et SD pour voir ou est l'erreur qui produit le NaN 
# on ignore ALT_sunshine_4 car cree un NaN en divisant avec SD=0
begin
    stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), select(weather_filled, Not(:precipitation_nextday))) 
    fit!(stan_mach, verbosity=2) 
    MLJ.transform(stan_mach, weather_filled)
end"""

begin
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

# Test data
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	#filled_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
    cleaned_test_data = dropmissing(test_data)
    #stan_test_data = MLJ.transform(stan_mach,cleaned_test_data)
end

# Apply multiple logistic regression
# With L2 regularization (Ridge) 
begin
	model = LogisticClassifier(penalty = :l2)
	#lambda = [[i/10 for i in 1:10];[j for j in 1:20]]
    lambda = [[10*i for i in 1:20];[j for j in 1:30];[k*10 for k in 4:10]]
	selftuning_lambda = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :lambda, values = lambda),
                                   measure = auc)
	selftuning_lambda_mach = fit!(machine(selftuning_lambda, input, output))
    # avec PCA, donne des résultats bien pire ...
    # selftuning_lambda_mach = fit!(machine(@pipeline(PCA(), selftuning_lambda), input,output))
end
confusion_matrix(predict_mode(selftuning_lambda_mach, input), output)

"""# Evaluation with AUC and confusion matrix on full input set
begin
    report(selftuning_lambda_mach)
    auc_val = MLJ.auc(predict(selftuning_lambda_mach,input),output)
    confusion_matrix(predict_mode(selftuning_lambda_mach, input), output)
end"""

# j'arrive pas a faire de confusion_matrix quand j'utilise le standardizer()
begin
	probs_L2 = MLJ.predict(selftuning_lambda_mach, stan_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "logistic_selftuned_L2_regression_stand_new.csv"), df_L2)
	# CSV.write(joinpath(@__DIR__, "..", "results", "logistic_selftuned_L2_regression_with_PCA.csv"), df_L2)
end