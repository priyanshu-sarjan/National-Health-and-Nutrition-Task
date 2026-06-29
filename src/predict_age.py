import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def main():
    print("--- Initializing NHANES Age Group Prediction Pipeline ---")
    
    # Correct relative paths dynamically based on execution point
    train_path = os.path.join(os.path.dirname(__file__), '../data/Train_dataset (1).csv')
    test_path = os.path.join(os.path.dirname(__file__), '../data/Test_dataset (1).csv')
    sub_output_dir = os.path.join(os.path.dirname(__file__), '../submissions')
    
    os.makedirs(sub_output_dir, exist_ok=True)

    # 1. Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"[INFO] Datasets successfully loaded. Train size: {train.shape}, Test size: {test.shape}")

    # 2. Extract targets and clean structures
    train['target'] = train['age_group'].apply(lambda x: 1 if x == 'Senior' else 0)
    X_train = train.drop(columns=['SEQN', 'age_group', 'target'])
    y_train = train['target']
    X_test = test.drop(columns=['SEQN'])

    # 3. Clean Missing Data Structures
    num_cols = ['BMXBMI', 'LBXGLU', 'LBXGLT', 'LBXIN']
    cat_cols = ['RIAGENDR', 'PAQ605', 'DIQ010']

    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
    print("[INFO] Imputation complete. Missing data addressed without leakage risks.")

    # 4. Cross Validation Safety Check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05)
    
    print("[INFO] Evaluating model stability via 5-Fold Stratified CV...")
    cv_scores = cross_val_score(final_model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"[RESULT] Local Cross-Validation Mean Accuracy Score: {cv_scores.mean():.4f}")

    # 5. Fit full training profile and predict
    print("[INFO] Executing comprehensive training pass over full dataset...")
    final_model.fit(X_train, y_train)
    test_predictions = final_model.predict(X_test)

    # 6. Generate submission profile deliverable
    submission_df = pd.DataFrame({'age_group': test_predictions})
    output_path = os.path.join(sub_output_dir, 'final_submission.csv')
    submission_df.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Target submission file generated at: '{output_path}'")
    print("--- Pipeline Execution Successfully Complete ---")

if __name__ == "__main__":
    main()
