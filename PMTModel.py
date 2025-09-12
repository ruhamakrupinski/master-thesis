import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as smf

real_hh_ind = [
    'H150104_TENURE_STA',
    'H150113_RADIO_QTY',
    'H150114_TV_QTY',
    'H150115_VCD_QTY',
    'H150116_STEREO_QTY',
    'H150117_REF_QTY',
    'H150118_WASH_QTY',
    'H150119_AIRCON_QTY',
    'H150121_LANDLINE_QTY',
    'H150122_CELL_QTY',
    'H150123_COMP_QTY',
    'H150124_OVEN_QTY',
    'H150120_CAR_QTY',
    'H150125_BANCA_QTY',
    'H150126_MOTOR_QTY',
    'H150102_ROOF',
    'H150103_WALLS',
    'H150101_BLDG_TYPE',
    'H150111_WATER_SUPPLY',
    'H150109_TOILET',
    'H150110_ELECTRICITY',
    #'HHTYPE',
    'domestic_helper',
    'RPROV',
    #'URB',
    #'AGIND'
    #'IND_4PS'
]

real_id_ind = [
    'FSIZE',
    'hh_sex',
    'hh_ms',
    'occ4d_1',
    'occ4d_2',
    'occ4d_3',
    'occ4d_4',
    'occ4d_5',
    'occ4d_6',
    'occ4d_7',
    'occ4d_8',
    'occ4d_9',
    'age_0_5',
    'age_13_17',
    'age_18_64',
    'age_65p',
    'age_6_12',
    'educ_college',
    'educ_none',
    'educ_other',
    'educ_postgrad',
    'educ_primary',
    'educ_secondary'
]

real_com_ind = [
    "poblacion", "street_pattern", "acc_nat_hwy", "dist_nat_hwy",
    "dist_city_hall", "dist_market", "dist_elem_sch", "dist_hs_sch",
    "dist_brgy_health", "dist_hospital", "dist_plaza", "waterworks",
    "cell_signal", "dist_landline", "dist_post_office", "dist_fire_station"

]


X_terms = real_hh_ind + real_id_ind + real_com_ind          # one flat list

cat = [
    "H150104_TENURE_STA", "H150102_ROOF", "H150103_WALLS", "H150101_BLDG_TYPE",
    "H150111_WATER_SUPPLY", "H150109_TOILET", "H150110_ELECTRICITY", "hh_sex", "hh_ms",
]
num = [c for c in X_terms if c not in cat]


class PMTModel:
    def __init__(self, train_df, logistic_regression_threshold=0.86, filtere_data=True):
        """We assume the DataFrame `df` has been preprocessed and contains the necessary columns.
        
        The y variable is assumed to be 'log_PCINC', and the independent variables.
        """
        self.train_df = train_df.copy()  # Avoid modifying the original DataFrame
        self.original_columns = train_df.columns.tolist()
        self.logistic_regression_threshold = logistic_regression_threshold

        print("Training Linear model...")
        self.linear_model = self._train_linear_model(train_df)
        # Add the predicted income to the training DataFrame
        pred = self.linear_model.get_prediction(train_df)

        # Predict over training data
        sf   = pred.summary_frame(alpha=0.05)       # 95 % level → z ≈ 1.96
        self.train_df["PCINC_LOWER_BOUND_18"] = np.exp(sf["obs_ci_lower"])
        # Get labels for poverty line
        self.train_df['POOR_PRE_18'] = self.train_df['PCINC_LOWER_BOUND_18'] <= self.train_df['poverty_line']
        self.train_df['POOR_18'] = self.train_df['PCINC'] <= self.train_df['poverty_line']
        # 0‑1 label for initial poor flag
        self.train_df["poor_s1"] = self.train_df["POOR_PRE_18"].astype(int)

        # Prepare the training DataFrame for logistic regression
        # restrict to flagged households + ground‑truth labels
        if filtere_data:
            filtered_df = self.train_df[self.train_df["poor_s1"] == 1].copy()
        else:
            filtered_df = self.train_df.copy()
        filtered_df["true_nonpoor"] = (filtered_df["PCINC"] > filtered_df["poverty_line"]).astype(int)
        filtered_df["true_poor"] = 1 - filtered_df["true_nonpoor"]

        # Train the logistic regression model
        print("Training Logistic Regression model...")
        self.logistic_model = self._train_logistic_regression(filtered_df)






    def _train_linear_model(self, df):
        # --- 1. choose the dependent variable ---------------------------------------
        y = "log_PCINC"              # exactly the column name in df_18

        # --- 2. build the RHS (X) part ----------------------------------------------
        # All the entries that already start with C( … ) are *already* wrapped, so we
        # leave them untouched.  Everything else we leave “as-is” because they’re numeric.

        X_terms = real_hh_ind + real_id_ind + real_com_ind          # one flat list
        rhs     = " + ".join(X_terms)                # → "C(H150104_TENURE_STA) + H150113_RADIO_QTY + …"

        # --- 3. stitch the full formula ---------------------------------------------
        formula = f"{y} ~ {rhs}"
        print("Formula being sent to Patsy:\n", formula)

        # --- 4. run the model --------------------------------------------------------
        model = smf.ols(formula, data=df).fit()

        return model
    
    def _train_logistic_regression(self, filtered_train_df):
        """Train a logistic regression model to predict poverty status."""


        pipeline = make_pipeline(
            make_column_transformer((OneHotEncoder(drop="first", handle_unknown="ignore"), cat), (StandardScaler(), num)),
            LogisticRegression(solver="saga", max_iter=100_000, n_jobs=-1),
        ).fit(filtered_train_df[cat + num], filtered_train_df["true_nonpoor"])

        return pipeline
    
    def predict_linear_regression(self, df):
        """Predict income for a new DataFrame."""
        # Ensure the DataFrame has the same columns as the training DataFrame
        for col in X_terms:
            if col not in df.columns:
                print(f"Warning: Input DataFrame is missing column: {col}")
                # raise ValueError(f"Input DataFrame is missing required column: {col}")

        # copy locally to avoid modifying the original DataFrame
        df = df.copy()

        # Predict using the linear model
        df["PCINC_LOWER_BOUND_18"] = np.exp(self.linear_model.get_prediction(df).summary_frame(alpha=0.05)["obs_ci_lower"])
        df['POOR_PRE_18'] = df['PCINC_LOWER_BOUND_18'] <= df['poverty_line']
        return df['POOR_PRE_18'].astype(int)
    
    def predict(self, df, threshold=None):
        """Predict poverty status for a new DataFrame."""
        if threshold is None:
            threshold = self.logistic_regression_threshold
        # Ensure the DataFrame has the same columns as the training DataFrame
        pred = self.predict_linear_regression(df)

        # 0‑1 label for initial poor flag
        df["poor_s1"] = pred

        # Predict using the logistic regression model
        df["prob_fp"] = self.logistic_model.predict_proba(df[cat + num])[:, 1]
        df["poor_final"] = np.where(
            df["poor_s1"] & (df["prob_fp"] < threshold),
            1, 0).astype(int)

        return df["poor_final"]
    
    def predict_proba_of_being_rich(self, df):
        """Predict poverty probabilities for a new DataFrame."""
        pred = self.predict_linear_regression(df)

        # 0‑1 label for initial poor flag
        df["poor_s1"] = pred

        # Predict using the logistic regression model
        df["prob_of_rich"] = self.logistic_model.predict_proba(df[cat + num])[:, 1]

        return df["prob_of_rich"]

