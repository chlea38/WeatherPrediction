# import packages
begin
    using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using Markdown
    using InteractiveUtils
    using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots, LinearAlgebra, Statistics, Random, CSV, MLJLinearModels, StatsBase
end

#Importing data
begin
    weather = CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame)
    coerce!(weather, :precipitation_nextday=>Multiclass)
    weather_filled = MLJ.transform(fit!(machine(FillImputer(), weather)), weather) #Filling imputer data during the fit 
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end


"""# Create a matrix from Dataframe
weather_m = Matrix(weather)"""

# Standardization
# n'est pas utilisé après car contient des Infs or NaNs
input_st = MLJ.transform(fit!(machine(Standardizer(), input)), input) 

# Using PCA to remove unecessary data
# utilise input et pas input_st parce que askip ca contient des "Infs or NaNs" (infinity number or Not a Number)
# du coup ca fait une erreur -> utilise input non transformé
mach_pca = fit!(machine(PCA(), input))
report(mach_pca)

#mach_PCA = fit!(machine(@pipeline(Standardizer(), PCA()), input))
mach2 = fit!(machine(@pipeline(PCA(), LogisticClassifier(penalty = :none)),input, output))
