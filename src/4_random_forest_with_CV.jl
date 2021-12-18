# This complete pipeline perform: 
# - filling missing data with FillImputer
# - training the machine with random forest classification/regression on training set 
# - validation of the machine on validation set
# - if good, finer fitting on full input set 
# - apply machine to test input and save prediction in a CSV file

# Import usefull packages
begin
    using Markdown, InteractiveUtils, Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using DataFrames, CSV, MLJ, OpenML, MLCourse, LinearAlgebra, Random, MLJLinearModels
    Pkg.add("MLJDecisionTreeInterface")
    using MLJDecisionTreeInterface
    Random.seed!(1)
end

# Import training & test data and deal with missing data with FillImputer
begin
	weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>OrderedFactor{2})
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) #Filling imputer data during the fit 
end

# Standardization
# contient des Infs or NaNs -> verbosity pour afficher les moyennes et SD pour voir ou est l'erreur qui produit le NaN 
# on ignore ALT_sunshine_4 car cree un NaN en divisant avec SD=0
begin
    stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), select(weather_filled, Not(:precipitation_nextday))) 
    fit!(stan_mach, verbosity=2) 
    MLJ.transform(stan_mach, weather_filled)
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

#coerce the multi class so that they have an ordering
#coerce!(output,binary)
#output = coerce(weather_filled.precipitation_nextday, OrderedFactor)

begin
    test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
    stan_test_data = MLJ.transform(stan_mach,cleaned_test_data)
end

# separate into training and validation sets to fit the machine 
# once we are satisfied with a result, fit the machine with the same parameters & hyperparameters on the whole set
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
    confusion_matrix(predict_mode(random_forest_class, val_input), val_output)
    mean(predict_mode(random_forest_class, val_input) .== val_output)
end

#random forest classification/regression on full input set for precision
begin
    random_forest_mach = fit!(machine(RandomForestClassifier(n_trees = 600), input, output))
    evaluate!(random_forest_mach, resampling=CV(), measure = auc)
    confusion_matrix(predict_mode(random_forest_mach, input), output)
    # mean(predict_mode(random_forest_reg, input) .== output)
end

pred = predict(random_forest_mach,train_input)

"""begin
	probs_train = MLJ.predict(random_forest_class, input).prob_given_ref.vals
	N = size(input)[1]
	df = DataFrame(id = 1:N, precipitation_nextday = probs_train[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "random_forest_reg_600_train.csv"), df)
end"""

#generation of test output file
begin
	probs = MLJ.predict(random_forest_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df2 = DataFrame(id = 1:N, precipitation_nextday = probs[2])
    CSV.write(joinpath(@__DIR__, "..", "results", "random_forest_600.csv"), df2)
end