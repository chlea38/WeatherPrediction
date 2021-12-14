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
# verbosity pour afficher les moyennes et SD pour voir ou est l'erreur qui produit le NaN 
# on ignore ALT_sunshine_4 car cree un NaN en divisant avec SD=0
begin
    stan_mach = machine(Standardizer(features = [:ALT_sunshine_4], ignore = true), input) 
    fit!(stan_mach, verbosity=2) 
    MLJ.transform(stan_mach, weather_filled)
    input = select(weather_filled, Not(:precipitation_nextday))
    output = weather_filled.precipitation_nextday
end

# Using PCA to remove unecessary data
# utilise input et pas input_st parce que askip ca contient des "Infs or NaNs" (infinity number or Not a Number)
# du coup ca fait une erreur -> utilise input non transformé
mach_pca = fit!(machine(PCA(), input))
report(mach_pca)

#mach_PCA = fit!(machine(@pipeline(Standardizer(), PCA()), input))
mach2 = fit!(machine(@pipeline(PCA(), LogisticClassifier(penalty = :none)),input, output))

"""#resultat NaN avec standardizer()
begin
    stan_mach = machine(Standardizer(), weather_filled)
    fit!(stan_mach, verbosity=2)
    weather_trained = MLJ.transform(stan_mach,weather_filled)
    weather_test = MLJ.transform(stan_mach,cleaned_test_data)
    confusion_matrix(predict_mode(stan_mach,input)output)
end

begin
	probs_stan = MLJ.predict(stan_mach, cleaned_test_data).prob_given_ref.vals
	N = size(cleaned_test_data)[1]
	df_L2 = DataFrame(id = 1:N, precipitation_nextday = probs_L2[2])
	confusion_matrix(predict_mode(stan_mach, input), output)
end"""