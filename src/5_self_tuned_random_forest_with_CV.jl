# This complete pipeline perform: 
# - filling missing data with FillImputer
# - training the machine with random forest classification/regression on training set 
# - selftuning for the number of tree
# - apply machine to test data and save prediction in a CSV file

# Import usefull packages
begin
    using Markdown, InteractiveUtils, Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using DataFrames, CSV, MLJ, MLJLinearModels, OpenML, MLCourse, MLJMultivariateStatsInterface, LinearAlgebra, Random, StatsBase
    Pkg.add("MLJDecisionTreeInterface")
    using MLJDecisionTreeInterface
    Random.seed!(1)
end

# Import training & test data and deal with missing data with FillImputer
begin
	weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>OrderedFactor{2})
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) 
end

"""# Standardization
# contient des Infs or NaNs -> verbosity pour afficher les moyennes et SD pour voir ou est l'erreur qui produit le NaN 
# on ignore ALT_sunshine_4 car cree un NaN en divisant avec SD=0
begin
    #stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), select(weather_filled, Not(:precipitation_nextday))) 
    #fit!(stan_mach, verbosity=2) 
    #MLJ.transform(stan_mach, weather_filled)
end"""

begin
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

#bizarre... meilleures resultats sans standardizer le test set...
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
    #filled_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
    #stan_test_data = MLJ.transform(stan_mach,cleaned_test_data)
end

#random forest classification with selftuned number of tree: takes a long time to run
begin
	model = RandomForestClassifier()
	n_trees = [[i*100 for i in 1:10];[j*1000 for j in 1:5]]
	selftuning_tree = TunedModel(model = model,
                                   resampling = CV(nfolds=5),
                                   tuning = Grid(),
                                   range = range(model, :n_trees, values = n_trees),
                                   measure = auc)
	selftuning_tree_mach = fit!(machine(selftuning_tree, input, output))
    confusion_matrix(predict_mode(selftuning_tree_mach, input), output)
end

begin
	probs = MLJ.predict(selftuning_tree_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "selftuned_random_forest.csv"), df)
end