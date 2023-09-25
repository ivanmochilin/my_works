# Databricks notebook source
artifactory_user = dbutils.secrets.get(scope="uapc-prj-kv-secret-scope", key="ArtifactoryTechUserName")
artifactory_api_key = dbutils.secrets.get(scope="uapc-prj-kv-secret-scope", key="ArtifactoryTechUserApiKey")
%pip install properscoring catboost yellowbrick shap numpy==1.23.5 pyspark==3.4.0 kaleido --upgrade-strategy "only-if-needed" --index-url https://$artifactory_user:$artifactory_api_key@schwarzit.jfrog.io/artifactory/api/pypi/pypi/simple

# COMMAND ----------

# MAGIC %md ## Parameters

# COMMAND ----------

try:
    WSHOP_CD = dbutils.widgets.get("wshop_cd")
except:
    WSHOP_CD = "DE"
    
MODEL_NAME = f"paf_promo_{WSHOP_CD.lower()}"
MODEL_NAME_ALL = "paf_promo_all_wshops"

experiment_dict = {
    "DE": "4428095351787503",
    "NL": "955027284186048",
    "CZ": "955027284186050",
    "PL": "955027284186051",
    "BE": "955027284186052",
    "SK": "955027284186047",
}

# mlflow tracking
EXPERIMENT_ID = experiment_dict[WSHOP_CD]

columns = [
    "sell_off_horizon",
    "promotion_medium_type",
    "brand_type_cd",
    "num_promo",
    "promo_date",
    "tv_fg",
    "decay_50_daily_sales",
    "store_sales_wt",
    "wshop_cd",
    "wt_avg_promo_type",
    "wt_diff_in_sales_price",
    "decay_50_wt_avg_sum_sales",
    "wt_avg_based_on",
    "decay_50_nl_avg_weekly_sales",
]
INPUT_COLUMNS = columns

TARGET = "target_nl" if "after" in MODEL_NAME else "target_wt"

TEST_RANGE_PLAN_CD = [2101, 2104, 2107, 2110]
#VAL_RANGE_PLAN_CD = [2104]
#LOG_TRANSFORM_TARGET = True

# COMMAND ----------

[print(i) for i in [WSHOP_CD, MODEL_NAME, EXPERIMENT_ID, TEST_RANGE_PLAN_CD, INPUT_COLUMNS, TARGET]]

# COMMAND ----------

# MAGIC %md ## Create training set

# COMMAND ----------

# MAGIC %run "/Repos/Shared/schwarzit.uapc-nfdf-paf/databricks/notebooks/input_prep_functions"

# COMMAND ----------

processor = DataProcessor(wshop_cd=None, coal_logic=True)
train_data_old = processor.load_preprocessed_data().toPandas()
predict_data = processor.load_preprocessed_predict_data(range_plan_cd=[2307]).toPandas()

# COMMAND ----------

params_production = {
    'random_state': 123,
    'iterations': 311,
    'depth': 6,
    'l2_leaf_reg': 0.2913787635862005,
    'learning_rate': 0.08719333293475111,
    'random_strength': 4.160392001940713,
    'colsample_bylevel': 0.4517702061073786,
    'bootstrap': {'bootstrap_type_c': 'Bayesian', 'bagging_temperature': 0}
}

workflow = ModelWorkflow(df=train_data_old,
                    model_name=MODEL_NAME_ALL,
                    wshop_cd=['DE', 'CZ', 'PL', 'NL', 'BE'],
                    input_columns=INPUT_COLUMNS,
                    test_range_plan_cd=TEST_RANGE_PLAN_CD,
                    loss_metric = 'RMSE',
                    confidence=0.9,
                    log_transform_target=True,
                    noise_detector=False,
                    batch_size = 250,
                    subsample = None,
                    overfit_error = 15,
                    params_det = params_production)

# COMMAND ----------

mlflow.end_run()


space_cat={
    'learning_rate': hp.uniform('learning_rate',0.001,0.1),
    'iterations': hp.quniform('iterations',200,600,1),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg',0.01,1),
    'depth': hp.quniform('depth',6,8,1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5,1),
    'random_strength': hp.uniform('random_strength', 0.001,10),
    'bootstrap': hp.choice('bt', [
        {'bootstrap_type_c': 'Bayesian', 'bagging_temperature': hp.uniform('bagging_temperature',0,10)},
        {'bootstrap_type_c': 'Bernoulli', 'subsample': hp.uniform('subsample', 0.1,1)},
        ])
    }


mlflow.start_run()
workflow.optimize_and_fit(space_cat = space_cat,
        max_evals = 70,
        iter_size = 30,
        n_forecasts = 10000,
        log_model = True,
        register_model = False,
        cv = 3)
mlflow.end_run()

workflow.best_params

# COMMAND ----------

try:
    mlflow.start_run()
except:
    mlflow.end_run()
    mlflow.start_run()
workflow.fit_regressor(params=params_production, iter_size=30, n_forecasts=10000, fit_on_all_data=False, log_model=True, register_model=True)
mlflow.end_run()

bins={
    "promo": [1199.038356,   2240.164473],
    "after_promo": [1199.038356,   2240.164473]
}
logged_model = 'runs:/61249277a6704d8f8ff1d4756241b502/paf_promo_all_wshops_model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
unwrapped_model = loaded_model.unwrap_python_model()
unwrapped_model.get_bins(bins)
loaded_model.predict(workflow.X_test[0:2000])

# COMMAND ----------

bins={
    "promo": [1199.038356,   2240.164473],
    "after_promo": [1199.038356,   2240.164473]
}

model = mlflow.pyfunc.load_model(f'models:/paf_promo_all_wshops/Production')
# model = mlflow.pyfunc.load_model('runs:/01ef5e7367f34b139c83ce8e6c5d4481/paf_promo_all_wshops_model')
unwrapped_model = model.unwrap_python_model()
unwrapped_model.get_bins(bins)

model.predict(workflow.X_test[0:2000])

# COMMAND ----------

 #inpt_cols = ["sell_off_horizon",
 #   "promotion_medium_type",
 #   "brand_type_cd",
 #   "item_name_embeddings_0",
 #   "item_name_embeddings_1",
 #   "item_name_embeddings_2",
 #   "tv_fg",
 #   "store_sales_wt",
 #   "wt_avg_based_on",
 #   "wt_avg_promo_type",
 #   "nl_avg_based_on"]
 #ist_dec = [25, 50, 75, 100]
 #or i in list_dec:
 #   try:
 #       inpt_cols.remove(dec_sales)
 #   except:
 #       pass
 #   if i == 100:
 #       dec_sales = "wt_avg_sum_sales"
 #   else:
 #       dec_sales = "decay_" + str(i) + "_wt_avg_sum_sales"
 #   inpt_cols.append(dec_sales)
 #   for j in list_dec:
 #       try:
 #           inpt_cols.remove(dec_price)
 #       except:
 #           pass
 #       if j == 100:
 #           dec_price = "wt_diff_in_sales_price"
 #       else:
 #           dec_price = "wt_diff_" + str(j) + "_in_sales_price"
 #       inpt_cols.append(dec_price)
 #       for k in list_dec:
 #           try:
 #               inpt_cols.remove(dec_sales_nl)
 #           except:
 #               pass
 #           if k == 100:
 #               dec_sales_nl = "nl_avg_weekly_sales"
 #           else:
 #               dec_sales_nl = "decay_" + str(k) + "_nl_avg_weekly_sales"
 #           inpt_cols.append(dec_sales_nl)
 #           for n in list_dec:
 #               try:
 #                   inpt_cols.remove(dec_daily)
 #               except:
 #                   pass
 #               if n == 100:
 #                   dec_daily = "daily_sales"
 #               else:
 #                   dec_daily = "decay_" + str(n) + "_daily_sales"
 #               inpt_cols.append(dec_daily)
 #               workflow = ModelWorkflow(df=train_data,
 #                   model_name=MODEL_NAME,
 #                   wshop_cd=WSHOP_CD,
 #                   input_columns=inpt_cols,
 #                   test_range_plan_cd=TEST_RANGE_PLAN_CD,
 #                   loss_metric = 'Huber',
 #                   confidence=0.9)
 #               try:
 #                   mlflow.start_run()
 #               except:
 #                   mlflow.end_run()
 #                   mlflow.start_run()
 #              workflow.fit_regressor(selected_model='catboost', params=None, delta=9.57, iter_size=15, n_forecasts=1000, fit_on_all_data=False, log_model=True, log_transform_target=True, register_model=False)
 #              mlflow.end_run()


# COMMAND ----------

#for batch_iter in range(20, 340, 20):
#    workflow = ModelWorkflow(df=train_data,
#                    model_name=MODEL_NAME,
#                    wshop_cd=WSHOP_CD,
#                    input_columns=INPUT_COLUMNS,
#                    test_range_plan_cd=TEST_RANGE_PLAN_CD,
#                    loss_metric = 'Huber',
#                    confidence=0.9,
#                    log_transform_target=True,
#                    batch_size=batch_iter)
#    try:
#        mlflow.start_run()
#    except:
#        mlflow.end_run()
#        mlflow.start_run()
#    workflow.fit_regressor( params=None, delta=10.77, iter_size=15, n_forecasts=1000, fit_on_all_data=False, log_model=True, register_model=False, remove_noise=True)
#    mlflow.end_run()
#    print(batch_iter)

# COMMAND ----------

#from mlflow.entities import ViewType
#from datetime import datetime, timedelta
#date_from = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
#query = 'tags.tuned = "False" and tags.wshop_cd = "DE" and metrics.model.test_mae <= 1316 and metrics.model.test_rmse <= 3200'
#runs = mlflow.MlflowClient().search_runs(
#    experiment_ids=all_experiments,
#    filter_string=query,
#    run_view_type=ViewType.ALL,
#    order_by=["attributes.created DESC"]
#)

# COMMAND ----------

#runs

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# try:
#     mlflow.end_run()
# except:
#     pass
# delta_space =  hp.uniform('delta', 1, 20)
# mlflow.start_run()
# workflow.find_best_huber_param(delta_space=delta_space,  max_evals=70, cv=3, log_model=True)
# mlflow.end_run()

# COMMAND ----------

#cat_features  = ["promotion_medium_type", "wt_avg_based_on", "wt_avg_promo_type", "tv_fg", "brand_type_cd"]
#emb_features =['embedding'] 
#pool = ctb.Pool(workflow.X_train, np.log1p(workflow.y_train) , cat_features = [workflow.X_train.columns.get_loc(col) for col in cat_features], embedding_features = [workflow.X_train.columns.get_loc(col) for col in emb_features])
#reg = ctb.CatBoostRegressor(allow_writing_files=False, random_state=123, loss_function='Huber:delta=9.57')
#reg.fit(pool)
#pool_test = ctb.Pool(workflow.X_test, np.log1p(workflow.y_test) , cat_features = [workflow.X_test.columns.get_loc(col) for col in cat_features], embedding_features = [workflow.X_test.columns.get_loc(col) for col in emb_features])
#preds = np.expm1(reg.predict(pool_test))
#mae = mean_absolute_error(preds, workflow.y_test)
#rmse =  mean_squared_error(preds, workflow.y_test, squared=False)
#print(mae, rmse)

# COMMAND ----------



# COMMAND ----------

# Comparison between models
#wshop_iter = ['DE', 'CZ','BE', 'PL', 'NL']
#input_columns_selected = [
#    "sell_off_horizon",
#    "promotion_medium_type",
#    "brand_type_cd",
#    "num_promo",
#    "promo_date",
#    "tv_fg",
#    "decay_50_daily_sales",
#    "store_sales_wt",
#    "wt_avg_promo_type",
#    "wt_diff_in_sales_price",
#    "decay_50_wt_avg_sum_sales",
#    "wt_avg_based_on",
#    "decay_50_nl_avg_weekly_sales",
#]
#input_columns_all_wshops = input_columns_selected.copy()
#input_columns_all_wshops.append("wshop_cd")
#for i in wshop_iter:  
#    processor = DataProcessor(wshop_cd=i, coal_logic=True)
#    train_data_selected = processor.load_preprocessed_data().toPandas()
#    workflow_all_wshops = ModelWorkflow(df=train_data_old,
#                    model_name=MODEL_NAME_ALL,
#                    wshop_cd=i,
#                    input_columns=input_columns_all_wshops,
#                    test_range_plan_cd=TEST_RANGE_PLAN_CD,
#                    loss_metric = 'RMSE',
#                    confidence=0.9,
#                    log_transform_target=True)
#    workflow_selected = ModelWorkflow(df=train_data_selected,
#                    model_name=MODEL_NAME_ALL,
#                    wshop_cd=i,
#                    input_columns=input_columns_selected,
#                    test_range_plan_cd=TEST_RANGE_PLAN_CD,
#                    loss_metric = 'RMSE',
#                    confidence=0.9,
#                    log_transform_target=True)
#    try:
#        mlflow.start_run()
#    except:
#        mlflow.end_run()
#        mlflow.start_run()
#    workflow_all_wshops.fit_regressor( params=None,  iter_size=30, n_forecasts=10000, fit_on_all_data=False, log_model=True, register_model=False, idx=None)
#    mlflow.end_run()
#
#    try:
#        mlflow.start_run()
#    except:
#        mlflow.end_run()
#        mlflow.start_run()
#    workflow_selected.fit_regressor( params=None,  iter_size=30, n_forecasts=10000, fit_on_all_data=False, log_model=True, register_model=False, idx=None)
#    mlflow.end_run()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

try:
    mlflow.start_run()
except:
    mlflow.end_run()
    mlflow.start_run()
workflow.fit_regressor(params=workflow.best_params, iter_size=30, n_forecasts=10000, fit_on_all_data=False, log_model=True, register_model=False)
mlflow.end_run()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

frontend_df = spark.createDataFrame(pd.DataFrame({
                            'item_ian_used_wt': [[331118, 353299, 312156]],
                            'item_ian_used_nl': [[282448, 331118, 353299]],
                            'range_plan_cd_wt': [[5.0, 3.0, 3.0]],
                            'range_plan_cd_nl': [[1.0, 2.0, 0]],
                            'tv_fg': [1],
                            'promotion_medium_type': ['HHZ'],
                            'sell_off_horizon': [52],
                            'wt_avg_based_on': [1],
                            'nl_avg_based_on': [1],
                            'price': [12],
                            'wshop_cd': ['DE'],
                            'item_ian': [439990],
                            'range_plan_cd_predict': [2307]
                            }))


# COMMAND ----------

frontend_df.createOrReplaceTempView("hui")

# COMMAND ----------

inference_2307 = spark.read.parquet('/mnt/paf/data/prediction_sets/inference_set_2307.parquet').toPandas()

# COMMAND ----------

# MAGIC %run "/Repos/ivan.mochilin@lidl.com/schwarzit.uapc-nfdf-paf/databricks/notebooks/input_prep_functions"

# COMMAND ----------

k=create_base_set(wshop='CZ')

# COMMAND ----------

pd.read_parquet('/dbfs/mnt/paf/data/prediction_sets/aggregated_sales_CZ.parquet')

# COMMAND ----------

inference = build_inference(frontend_df, WSHOP_CD)
inference

# COMMAND ----------

inference_2307.filter('item_ian == 439990').toPandas()

# COMMAND ----------

logged_model = 'runs:/39bcc9f0b2474009a84494e2d64bc832/paf_promo_de_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
#loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------

# MAGIC %md
# MAGIC Tv_fg analysis

# COMMAND ----------

input_tv = workflow.X_test.query('(tv_fg ==1) and (promotion_medium_type == 1)')
input_no_tv = input_tv.copy()
input_no_tv['tv_fg'] = 0

# COMMAND ----------

feature_prod = ['sell_off_horizon', 'promotion_medium_type', 'brand_type_cd', 'item_name_embeddings_0', 'item_name_embeddings_1', 'item_name_embeddings_2', 'tv_fg', 'daily_sales', 'store_sales_wt', 'wt_avg_sum_sales', 'wt_avg_based_on', 'wt_avg_promo_type', 'nl_avg_weekly_sales', 'wt_diff_in_sales_price']
input_prod_tv = train_data.query(f'(tv_fg == 1) and (promotion_medium_type == 1) and (range_plan_cd  in {TEST_RANGE_PLAN_CD})')[feature_prod]
input_prod_tv = input_prod_tv.rename(columns={'wt_diff_in_sales_price': 'diff_in_sales_price'})
input_prod_no_tv = input_prod_tv.copy()
input_prod_no_tv['tv_fg'] = 0

# COMMAND ----------

prod_model= 'runs:/44de7ab0995842c5aa1b3080ea3a1877/paf_promo_de_model'
real_prod = 'runs:/1cb95df1b74d4a54b35e688611b66560/paf_promo_de_model'
prod = mlflow.pyfunc.load_model(prod_model)
new_model = 'runs:/6d9a1d20a9f3487fb8a697ec7ec96dda/paf_promo_de_model'
new_mod = mlflow.pyfunc.load_model(new_model)
real_production = mlflow.catboost.load_model(real_prod)
prediction_real_prod_tv = np.exp(pd.DataFrame(real_production.predict(input_prod_tv)))
prediction_real_prod_no_tv = np.exp(pd.DataFrame(real_production.predict(input_prod_no_tv)))
prediction_prod_tv = pd.DataFrame(prod.predict(input_tv))
prediction_new_tv = pd.DataFrame(new_mod.predict(input_tv))
prediction_prod_no_tv = pd.DataFrame(prod.predict(input_no_tv))
prediction_new_no_tv = pd.DataFrame(new_mod.predict(input_no_tv))

# COMMAND ----------

prediction_real_prod_tv.agg(['mean', 'std'])

# COMMAND ----------

prediction_real_prod_no_tv.agg(['mean', 'std'])

# COMMAND ----------

st.ttest_ind(np.array(prediction_real_prod_tv), np.array(prediction_real_prod_no_tv), alternative='greater', equal_var=False)

# COMMAND ----------

st.mannwhitneyu(np.array(prediction_real_prod_tv), np.array(prediction_real_prod_no_tv), alternative='greater')

# COMMAND ----------

prediction_prod_tv.agg(['mean', 'std'])

# COMMAND ----------

prediction_new_tv.agg(['mean', 'std'])

# COMMAND ----------

prediction_prod_no_tv.agg(['mean', 'std'])

# COMMAND ----------

prediction_new_no_tv.agg(['mean', 'std'])

# COMMAND ----------

st.ttest_ind(prediction_prod_tv['wt_point_prediction'], prediction_new_tv['wt_point_prediction'], alternative='greater', equal_var=False)

# COMMAND ----------

st.ttest_ind(prediction_new_tv['wt_point_prediction'], prediction_new_no_tv['wt_point_prediction'], alternative='greater', equal_var=False)

# COMMAND ----------

st.mannwhitneyu(prediction_new_tv['wt_point_prediction'], prediction_new_no_tv['wt_point_prediction'], alternative='greater')

# COMMAND ----------

st.ttest_ind(prediction_prod_tv['wt_point_prediction'], prediction_prod_no_tv['wt_point_prediction'], alternative='greater', equal_var=False)

# COMMAND ----------

st.mannwhitneyu(prediction_prod_tv['wt_point_prediction'], prediction_prod_no_tv['wt_point_prediction'], alternative='greater')
