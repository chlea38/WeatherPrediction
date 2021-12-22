# This pipeline performs: 
# - filling missing data with FillImputer
# - standardize data
# - training the machine with XGBoost and selftuning the 3 hyperparameters
# - apply machine to test data and save prediction in a CSV file

# Import usefull packages
begin
    using Markdown, Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    Pkg.add("MLJXGBoostInterface")
    using  MLJ, MLJXGBoostInterface, DataFrames, CSV, MLCourse, MLJMultivariateStatsInterface, Statistics
end

# Import training & test data and deal with missing data with FillImputer
begin
	weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>OrderedFactor{2})
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) 
end

# Standardization
begin
    stand_mach = fit!(machine(Standardizer(), weather_filled))
    weather_filled_transf = MLJ.transform(stand_mach, weather_filled)
end

# Defining input & output
begin
    input = select(weather_filled_transf, Not(:precipitation_nextday))
    output = weather_filled_transf.precipitation_nextday
end

# Clean test data
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
    cleaned_test_data_transf = MLJ.transform(stand_mach,cleaned_test_data)
end

# XGBoost regressor 
begin
	model = XGBoostClassifier()
    eta = [0.01,0.03,0.06,0.1]
    num_round = 700:100:1000
    selftuning_XGBoost = TunedModel(model = model,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(),
                            range = [range(model, :eta, values = eta),
                                     range(model, :num_round, values = num_round),
                                     range(model, :max_depth, lower = 4, upper = 6)],
                                     measure = auc)
    selftuning_XGBoost_mach = fit!(machine(selftuning_XGBoost, input, output))
end

#inspect the results of selftuning to improve parameter ranges
fitted_params(selftuning_XGBoost_mach).best_model

# Evaluation with AUC and confusion matrix on full input set
report(selftuning_XGBoost_mach).best_model
report(selftuning_XGBoost_mach).best_history_entry.measurement
confusion_matrix(predict_mode(selftuning_XGBoost_mach, input), output)

begin
	probs = MLJ.predict(selftuning_XGBoost_mach, cleaned_test_data_transf).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "selftuned_XGBoost_regression_train_test_stan.csv"), df)
end
