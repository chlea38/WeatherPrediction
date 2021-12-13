# import packages
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using Markdown
using InteractiveUtils
using DataFrames, CSV, MLJ, MLJLinearModels, OpenML, NearestNeighborModels, Plots


# Training data
weather = dropmissing(CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame))

 # separate training data into training and validation sets
function partitionTrainValidation(data, at = 0.5) 
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    val_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[val_idx,:]
end

begin
	train,validation = partitionTrainValidation(weather,0.7)
	train_input = select(train, Not(:precipitation_nextday)) #ajouter le coerce! ou non?
	val_input = select(validation, Not(:precipitation_nextday))
end

# test data
begin
	test_data = CSV.read(joinpath(@__DIR__, "..",  "data", "testdata.csv"), DataFrame)
	cleaned_test_data = dropmissing(test_data)
end

# KNN Machine with K = 40
begin
	val_output = coerce!(validation, :precipitation_nextday=>Multiclass)
	train_output = coerce!(train, :precipitation_nextday=>Multiclass)
	KNN_mach = fit!(machine(KNNClassifier(K=40), val_input, 										validation.precipitation_nextday)) #fitting with validation
end

begin
	pred_train = predict(KNN_mach,train_input).prob_given_ref.vals #predicting on training
	rmse(pred_train[2],train_output[!,2])
end

# Output file
begin
	probs = predict(KNN_mach, cleaned_test_data).prob_given_ref.vals
	P = size(cleaned_test_data)[1]
	df_manual = DataFrame(id = 1:P, precipitation_nextday = probs[2])
	CSV.write(joinpath(@__DIR__, "..", "results", "KNN_manual_tuning.csv"), df_manual)
end


# Self-tuning KNN machine
begin 
	model_KNN = KNNClassifier()
	selftuning_KNN = TunedModel(model = model_KNN,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model_KNN, :K, values = 1:50),
                                   measure = auc)
	selftuning_KNN_mach = machine(selftuning_KNN, val_input, validation.precipitation_nextday) |> fit!
end

"""
#begin 
	#model_KNN_RR = KNNClassifier(regressor = RidgeRegressor())
	#selftuning_KNN_RR = TunedModel(model = model_KNN,
                                   #resampling = CV(nfolds = 10),
                                   #tuning = Grid(),
                                   #range = [range(model_KNN, :K, values = 1:50), range(model, :(regressor.lambda),
                                                  #lower = 1e-12, upper = 1e-3,
                                                  #scale = :log)],
                                   #measure = auc)
	#selftuning_KNN_RR_mach = machine(selftuning_KNN, val_input, validation.precipitation_nextday) |> fit!
#end

begin
	rep = report(selftuning_KNN_mach)
	# println(rep)
	best_KNN_mach = machine(KNNRegressor(K = 9), train_input, train.precipitation_nextday) |> fit!
end

scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "K", ylabel = "AUC")
"""

# Output file
begin
	pred_test = predict(selftuning_KNN_mach, cleaned_test_data).prob_given_ref.vals
	df_self = DataFrame(id = 1:P, precipitation_nextday = pred_test[2])
	CSV.write(joinpath(@__DIR__, "..", "results", "KNN_self_tuning.csv"), df_self)
end

