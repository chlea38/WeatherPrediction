using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using Markdown
using InteractiveUtils
using PlutoUI,DataFrames, CSV, Plots, MLJ, MLJLinearModels, OpenML, StatsPlots, MLCourse

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
	model_RR = RidgeRegressor() #serie 6
	selftuning_RR = TunedModel(model = model_RR,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(goal = 100),
                                   range = range(model_RR, :lambda,scale = :log,lower = 1e-10, upper = 1e-3),
                                   measure = rmse)
	selftuning_RR_mach = machine(selftuning_RR, val_input, validation.precipitation_nextday) |> fit!
end

#L'evaluation du model est fait en fonction du AUC sur kaggle, mais je n'arrive pas a faire fonctionner 
#cette machine avec le AUC mais que le RMSE. Apres il y a encore un probleme avec les types