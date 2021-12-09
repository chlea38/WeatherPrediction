using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using Markdown
using InteractiveUtils
using PlutoUI,DataFrames, CSV, Plots, MLJ, MLJLinearModels, OpenML, StatsPlots, MLCourse
import GLMNet: glmnet

PlutoUI.TableOfContents(title="Table of contents")

begin
    weather = dropmissing(CSV.read(joinpath(@__DIR__, "..",  "data", "trainingdata.csv"), DataFrame))
    weather_input = select(weather, Not(:precipitation_nextday)) 
    weather_output = coerce!(weather, :precipitation_nextday=>Multiclass)
    weather_fits = glmnet(Array(weather_input), weather_output)
end