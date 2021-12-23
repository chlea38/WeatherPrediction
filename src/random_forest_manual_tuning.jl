# This complete pipeline perform: 
# - filling missing data with FillImputer
# - train the machine with PCA() and Logistic Classifier, with L2 regularization as penalty
# (lambda is tuned by optimizing AUC)
# - save prediction in a CSV file

# Import usefull packages
begin
    using Markdown, InteractiveUtils, Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using DataFrames, CSV, MLJ, MLJLinearModels, OpenML, MLCourse, MLJMultivariateStatsInterface, LinearAlgebra, Random, StatsBase, MLJDecisionTreeInterface
    Random.seed!(1)
end

# Import training & test data and deal with missing data with FillImputer
begin
    weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>OrderedFactor{2})
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) #Filling imputer data during the fit 
end

# Standardization
begin
    stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), select(weather_filled, Not(:precipitation_nextday))) 
    fit!(stan_mach, verbosity=2) 
    MLJ.transform(stan_mach, weather_filled)
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

# Clean test data
begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = MLJ.transform(fit!(machine(FillImputer(), test_data)), test_data)
    cleaned_test_data_transf = MLJ.transform(stan_mach,cleaned_test_data)
end

# separate into training and validation sets to fit the machine 
function partitionTrainValidation(data, at = 0.5) 
    n = nrow(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    val_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[val_idx,:]
end

begin
    train,validation = partitionTrainValidation(weather_filled,0.5)
    train_input = select(train, Not(:precipitation_nextday)) 
    train_output = train.precipitation_nextday
    val_input = select(validation, Not(:precipitation_nextday))
    val_output = validation.precipitation_nextday 
end

#random forest classification on training set and prediction on validation set
begin
    random_forest_class = fit!(machine(RandomForestClassifier(n_trees = 600), train_input, train_output))
    evaluate!(random_forest_class, resampling=CV(), measure = auc)
    mean(predict_mode(random_forest_class, val_input) .== val_output)
end

#generation of validation output file
begin
    probs_train = MLJ.predict(random_forest_class, val_input)
    N = size(input)[1]
    df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_train[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "random_forest_reg_600_train.csv"), df_L2)
    confusion_matrix(predict_mode(random_forest_class, input), output)
end

#random forest classification/regression on full input set for precision
begin
    random_forest_mach = fit!(machine(RandomForestClassifier(n_trees = 600), input, output))
    evaluate!(random_forest_mach, resampling=CV(), measure = auc)
    confusion_matrix(predict_mode(random_forest_mach, input), output)
    mean(predict_mode(random_forest_mach, input) .== output)
end

#generation of test output file
begin
    probs = MLJ.predict(random_forest_mach, cleaned_test_data)
    N = size(cleaned_test_data)[1]
    df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "random_forest_reg_600.csv"), df_L2)
end
