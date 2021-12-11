using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using Markdown
using InteractiveUtils
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels, StatsBase, MLJModels

#Importing data
weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
input = select(weather, Not(:precipitation_nextday))
output = weather.precipitation_nextday

#Scaling the data matters! standardize(ZScoreTransform,data)
typeof(input)
weather_m = Matrix(weather)
stand_model = Standardizer()
transform(fit!(machine(stand_model, weather_m)), weather) 

#il n'aime pas le fait que weather ce soit un Dataframe, mais meme avec la matrice c'est bizarre
#peut-etre qu'il faut d'abord remplacer ce qui manque avec le pca et fillimputer
weather_stand = Standardizer(weather_m) 
mach_stan = standardize(ZScoreTransform,Base.AbstractVecOrMat(weather_m))

#Using PCA to remove unecessary data
mach_pca = machine(PCA(), weather)
fit!(mach_pca, rows=weather)
#fit!(mach_pca(FillImputer(), weather), rows=weather)

#Filling imputer data (during hte fit)
MLJ.transform(fit!(machine(FillImputer(), weather)), weather)

#mach_PCA = fit!(machine(@pipeline(Standardizer(), PCA()), input))
#mach_PCA = fit!(machine(@pipeline(Standardizer(), FillImputer(),PCA()), input))

logistic_mach = fit!(machine(FillImputer(LogisticClassifier(penalty = :none), input, output)))
