# This pipeline performs: 
# - filling missing data with FillImputer
# - training the machine with random forest classification on training set 
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

begin
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
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
end

# Evaluation with AUC and confusion matrix on full input set
report(selftuning_tree_mach).best_model
report(selftuning_tree_mach).best_history_entry.measurement
confusion_matrix(predict_mode(selftuning_tree_mach, input), output)

# save data in CSV file
begin
	probs = MLJ.predict(selftuning_tree_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "selftuned_random_forest.csv"), df)
end