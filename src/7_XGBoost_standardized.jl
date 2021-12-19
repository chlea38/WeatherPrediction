# This pipeline perform: 
# - filling missing data with FillImputer
# - standardize data
# - training the machine with XGBoost on training set and selftuning the 3 hyperparameters
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
# contient des NaNs car division par std = 0 
# -> on parcourt les std et on enlève les colonnes ou c'est 0

begin
    """N = size(weather_filled)
    param = fitted_params(fit!(machine(Standardizer(), weather_filled)))
    ignore = []
    for (key,tuple) in param
        if tuple[2] < 1
            push!(ignore, key)
        end
    end"""
    stand_mach = fit!(machine(Standardizer(), weather_filled))
    weather_filled_transf = MLJ.transform(stand_mach, weather_filled)
end

begin
    input = select(weather_filled_transf, Not(:precipitation_nextday))
    output = weather_filled_transf.precipitation_nextday
end

# clean test data
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
    cleaned_test_data_transf = MLJ.transform(stand_mach,cleaned_test_data)
end

# XGBoost regressor 
begin
	model = XGBoostClassifier()
    eta = [0.01,0.05,0.1,0.5]
    num_round = 300:100:600
    selftuning_XGBoost = TunedModel(model = model,
                            resampling = CV(nfolds = 4),
                            tuning = Grid(),
                            range = [range(model, :eta, values = eta),
                                     range(model, :num_round, values = num_round),
                                     range(model, :max_depth, lower = 3, upper = 6)],
                                     measure = auc)
    selftuning_XGBoost_mach = fit!(machine(selftuning_XGBoost, input, output))
    confusion_matrix(predict_mode(selftuning_XGBoost_mach, input), output)
end

begin
	probs = MLJ.predict(selftuning_XGBoost_mach, cleaned_test_data_transf).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "selftuned_XGBoost_regression_standardized.csv"), df)
end