| model | group | accuracy | macro_precision | macro_recall | macro_f1 | train_time_s | infer_ms_per_img |
|---|---|---|---|---|---|---|---|
| RandomForest | C classic | 0.922 | 0.917 | 0.894 | 0.904 | 59.3 | 0.056 |
| MLP_features | D MLP | 0.918 | 0.902 | 0.898 | 0.899 | 64.2 | 0.024 |
| LogisticRegression | C classic | 0.886 | 0.866 | 0.855 | 0.859 | 2.5 | 0.001 |
| MLP_pixels | D MLP | 0.885 | 0.861 | 0.848 | 0.853 | 669.3 | 0.230 |
| ClassGMM | B generative | 0.835 | 0.811 | 0.789 | 0.799 | 1.1 | 0.006 |
| KNN | C classic | 0.847 | 0.813 | 0.792 | 0.799 | 0.0 | 0.127 |
| DecisionTree | C classic | 0.812 | 0.777 | 0.771 | 0.773 | 48.2 | 0.002 |
| GaussianNB | C classic | 0.698 | 0.659 | 0.708 | 0.672 | 0.2 | 0.036 |
| GaussianNB_generative | B generative | 0.696 | 0.658 | 0.705 | 0.670 | 0.3 | 0.061 |
| GaussianNB_sameDims | B generative | 0.652 | 0.617 | 0.649 | 0.621 | – | – |
| DecisionTree_depth5 | C classic | 0.656 | 0.520 | 0.527 | 0.505 | – | – |
| DecisionTree_depth3 | C classic | 0.535 | 0.289 | 0.364 | 0.319 | – | – |
