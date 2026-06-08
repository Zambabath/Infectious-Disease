import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
#not using anymore
#from sklearn.linear_model import LogisticRegression

#many errors occured due to the low number of positive testing values which we cannot fully fix 
#so we use this to not see the errors so to keep the terminal clean. 
#If better data is acquired, this can be removed
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=HessianInversionWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

#the data
df = pd.read_csv("Project\ictrp_data - Copy.csv")
df2 = pd.read_csv("Project\ictrp_data_preprocessed_prim.csv")
y = df2["results_binary"]

#columns to drop which are either the target column, irrelevant columns, string columns, and other diseases
#Change which disease you want here and at the variable "Selected" in the loop
X_full = df2.drop(columns=["results_binary", "durations_day","country_codes",
                           "inclusion_age_max", "inclusion_age_min", "target_sample_size", "Soil-Transmitted Helminthiases", 
                           "Visceral Leishmaniasis",
                           "Schistosomiasis"])

#kfold and smote setup
#smote balances the training data so that the model can actually run
kf = KFold(n_splits=5, shuffle=True, random_state=42)
smi = SMOTE(random_state=42)


#lists to store the performance metrics and selected features from each fold for mean to be taken at end
metrics = {
    'Alpha': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Selected_Features': []
}

#start loop for each fold using the test train data
for fold, (train_data, test_data) in enumerate(kf.split(X_full, y)):
    #split data for the current fold
    X_train, X_test = X_full.iloc[train_data], X_full.iloc[test_data]
    y_train, y_test = y.iloc[train_data], y.iloc[test_data]
    
    #spply smote
    X_train_res, y_train_res = smi.fit_resample(X_train, y_train)

    #select the disease you want that will be guaranteed to be in the final model
    selected = ['Chagas Disease']
    remaining = [f for f in X_train_res.columns if f not in selected]
    
    #make a baseline aic to compare the next aics with - only uses disease itself
    X_start = sm.add_constant(X_train_res[selected])
    try:
        current_best_aic = sm.GLM(y_train_res, X_start, family=sm.families.Binomial()).fit(disp=0).aic
    except Exception:
        continue

    #start the aic loop for choosing the best aic using forward selection
    while remaining:
        best_aic_this_round = current_best_aic
        best_candidate_this_round = None
        
        #try each variable and see if the aic is better - chooses the best aic
        for candidate in remaining:
            test_features = selected + [candidate]
            X = sm.add_constant(X_train_res[test_features])
            
            try:
                model = sm.GLM(y_train_res, X, family=sm.families.Binomial())
                result = model.fit(disp=0, method="newton", maxiter=1000)
                
                if result.aic < best_aic_this_round:
                    best_aic_this_round = result.aic
                    best_candidate_this_round = candidate
                    
            except Exception:
                continue

        #choose best aic this round and either add to the model or that model is the best model - breaks at best model
        if best_candidate_this_round is not None:
            selected.append(best_candidate_this_round)
            remaining.remove(best_candidate_this_round)
            current_best_aic = best_aic_this_round
        else:
            break

    #gets the final model for confusion matrix and OR plot
    print(selected)
    metrics['Selected_Features'].append(selected)

    alpha = [0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000]
    alpha_results = []

    for i in alpha:
        X_final = sm.add_constant(X_train_res[selected])
        final_model = sm.GLM(y_train_res, X_final, family=sm.families.Binomial()).fit_regularized(
            method = "elastic_net",
            alpha = i,
            L1_wt = 0.0
        )
        #sklearn linearregression model - don't use since glm now works
        #final_model = LogisticRegression(
            #penalty='l2', 
            #C=0.1,
            #solver='liblinear', 
            #random_state=42
        #)
        #final_model.fit(X_train_res[selected], y_train_res)

        #predict probabilities
        #y_pred_prob = final_model.predict_proba(X_test_fold[selected])[:, 1]
        X_test_final = sm.add_constant(X_test[selected], has_constant='add')
        y_pred_prob = final_model.predict(X_test_final)
        #convert probabilities to binary using the threshold value 
        #try different values for different confusion matrix - top right high means you lower the value
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        #metric calcs
        current_f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        #store all metrics for the current alpha so they can be differentiated against
        alpha_results.append({
            'alpha': i,
            'Accuracy': accuracy_score(y_test, y_pred_binary),
            'Precision': precision_score(y_test, y_pred_binary, zero_division=0), 
            'Recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'F1 Score': current_f1
        })
        
    #convert alpha_results to a DataFrame and find the best alpha based on the F1 Score
    alpha_df = pd.DataFrame(alpha_results)
    
    #find the row with the best F1 Score
    best_row = alpha_df.loc[alpha_df['F1 Score'].idxmax()]
    
    #store in metrics the best metrics calculated
    metrics['Alpha'].append(best_row['alpha'])
    metrics['Accuracy'].append(best_row['Accuracy'])
    metrics['Precision'].append(best_row['Precision'])
    metrics['Recall'].append(best_row['Recall'])
    metrics['F1 Score'].append(best_row['F1 Score'])
    
    print(f"Fold {fold + 1} Best Alpha: {best_row['alpha']:.1e} | Best F1 Score: {best_row['F1 Score']:.4f}")

#calculate mean and standard deviation for all metrics calculated - gives a range of values and conf interval
for metric_name, values in metrics.items():
    if metric_name != 'Selected_Features':
        mean_score = np.mean(values)
        std_score = np.std(values)
        print(f"Average {metric_name}:  {mean_score:.4f} (±{std_score:.4f})")
    

feature_counts = Counter(feature for sublist in metrics['Selected_Features'] for feature in sublist)

#collects the features that were in at least 4 out of the 5 models
stable_features = [feature for feature, count in feature_counts.items() if count >= 0.8 * 5]
print(f"Features in at least 80% of the models: {stable_features}")

#makes the final model
X_confusion = sm.add_constant(X_full[stable_features])
final_best_model = sm.GLM(y, X_confusion, family=sm.families.Binomial()).fit()
print(final_best_model.summary())

#confusion matrix 
cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_binary)
cm_display.ax_.set_title(f"Confusion Matrix: Fold {5} Model (Threshold: {0.5})")
plt.show()

#odds ratio from here on out
#drops constant from model since that does not work with OR plot
params = final_best_model.params.drop('const')
conf_int_df = final_best_model.conf_int().drop('const') 

#OR and CI values
odds_ratios = np.exp(params)
odds_ratios_ci = np.exp(conf_int_df)

#gets lower and upper conf ints
lower_ci = odds_ratios_ci.columns[0]
upper_ci = odds_ratios_ci.columns[1]

#calculate error bars
error_bars = [
    (odds_ratios - odds_ratios_ci[lower_ci]),
    (odds_ratios_ci[upper_ci] - odds_ratios) 
]

plt.figure(figsize=(10, len(odds_ratios)))
plt.errorbar(odds_ratios, odds_ratios.index, 
              xerr=error_bars, 
              fmt='o',
              capsize=5)

plt.xscale('log')
#used for very large values to see how big they are and whether they should be removed - only used in testing
#plt.xlim(1e-100, 1e12)
plt.xlabel("Odds Ratio (OR)")
plt.ylabel("Selected Features")
plt.title("Odds Ratios Plot for Chagas Disease")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

odds_ratio_df = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    'Lower 95% CI': odds_ratios_ci.iloc[:, 0],
    'Upper 95% CI': odds_ratios_ci.iloc[:, 1]
}).sort_values(by='Odds Ratio', ascending=False)

print(odds_ratio_df)