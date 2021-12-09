using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using Markdown
using InteractiveUtils
using PlutoUI,DataFrames, CSV, Plots, MLJ, MLJLinearModels, OpenML, StatsPlots, MLCourse
import MLCourse: PolynomialRegressor, poly
import MLCourse: RidgeRegressor, ridge

PlutoUI.TableOfContents(title="Table of contents")

weather = dropmissing(CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame))

function partitionTrainValidation(data, at = 0.5) 
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    val_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[val_idx,:]
end

begin
	train,validation = partitionTrainValidation(weather,0.7)
	train_input = select(train, Not(:precipitation_nextday)) 
	val_input = select(validation, Not(:precipitation_nextday))
end

begin
	val_output = coerce!(validation, :precipitation_nextday=>Multiclass)
	train_output = coerce!(train, :precipitation_nextday=>Multiclass)
end

begin
    model_RR = PolynomialRegressor(regressor = RidgeRegressor()) #serie 6
    selftuning_model_RR = TunedModel(model = model_RR,
                                   tuning =  Grid(goal = 500),
                                   resampling = CV(nfolds = 5),
                                   range = [range(model, :degree, lower = 1, upper = 20),
                                            range(model, :(regressor.lambda),lower = 1e-12, upper = 1e-3, scale = :log)],
                                   measure = rmse)
    selftuning_mach_RR = machine(selftuning_KNN, val_input, validation.precipitation_nextday) |> fit!
end

begin
	rep = report(selftuning_mach_RR)
	best_KNN_mach = machine(RidgeRegressor(), train_input, train.precipitation_nextday) |> fit!
end

