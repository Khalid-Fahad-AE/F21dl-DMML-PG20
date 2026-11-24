| model | group | accuracy | macro_precision | macro_recall | macro_f1 | train_time_s | infer_ms_per_img |
|---|---|---|---|---|---|---|---|
| CNN_scratch_none_128 | E CNN | 0.996 | 0.995 | 0.995 | 0.995 | 918.2 | 2.041 |
| CNN_scratch_geo_128 | E CNN | 0.994 | 0.993 | 0.993 | 0.993 | 995.7 | 1.653 |
| TL_efficientnet_b0 | F transfer | 0.993 | 0.992 | 0.991 | 0.991 | 1235.5 | 2.676 |
| CNN_scratch_full_128 | E CNN | 0.992 | 0.989 | 0.992 | 0.990 | 1307.9 | 2.039 |
| TL_resnet50 | F transfer | 0.990 | 0.988 | 0.988 | 0.988 | 1447.6 | 3.276 |
| TL_mobilenet_v2 | F transfer | 0.986 | 0.983 | 0.985 | 0.984 | 1283.5 | 3.311 |
| RandomForest | C classic | 0.922 | 0.917 | 0.894 | 0.904 | 59.3 | 0.056 |
| MLP_features | D MLP | 0.918 | 0.902 | 0.898 | 0.899 | 64.2 | 0.024 |
| TL_efficientnet_b0_frozen | F transfer | 0.885 | 0.877 | 0.860 | 0.862 | 379.9 | – |
| LogisticRegression | C classic | 0.886 | 0.866 | 0.855 | 0.859 | 2.5 | 0.001 |
| MLP_pixels | D MLP | 0.885 | 0.861 | 0.848 | 0.853 | 669.3 | 0.230 |
| TL_mobilenet_v2_frozen | F transfer | 0.874 | 0.859 | 0.831 | 0.840 | 369.0 | – |
| TL_resnet50_frozen | F transfer | 0.861 | 0.862 | 0.822 | 0.831 | 418.2 | – |
| ClassGMM | B generative | 0.835 | 0.811 | 0.789 | 0.799 | 1.1 | 0.006 |
| KNN | C classic | 0.847 | 0.813 | 0.792 | 0.799 | 0.0 | 0.127 |
| DecisionTree | C classic | 0.812 | 0.777 | 0.771 | 0.773 | 48.2 | 0.002 |
| GaussianNB | C classic | 0.698 | 0.659 | 0.708 | 0.672 | 0.2 | 0.036 |
| GaussianNB_generative | B generative | 0.696 | 0.658 | 0.705 | 0.670 | 0.3 | 0.061 |
| GaussianNB_sameDims | B generative | 0.652 | 0.617 | 0.649 | 0.621 | – | – |
| DecisionTree_depth5 | C classic | 0.656 | 0.520 | 0.527 | 0.505 | – | – |
| DecisionTree_depth3 | C classic | 0.535 | 0.289 | 0.364 | 0.319 | – | – |
