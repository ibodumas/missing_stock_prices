"""
Interpolation Models:
The algorithms TENDS to perfectly fit the data. It doesn't generalize well, thus it might over-fit.
"""

heatmap(
    grid_search_result.Train_MSE,
    degree_spline,
    smoothing_factor,
    "Heatmap of Training MSE",
)
heatmap(
    grid_search_result.Validation_MSE,
    degree_spline,
    smoothing_factor,
    "Heatmap of Validation MSE",
)
