# Databricks notebook source
import mlflow
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.models.utils import ModelInputExample

import catboost as ctb
from databricks import feature_store
from pyspark.sql.functions import explode, array_union, when, col, arrays_zip
from pyspark.sql.types import ArrayType, FloatType, IntegerType
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

from sklearn import metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    make_scorer
)
import scipy.stats as st

from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit
import properscoring as ps

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

import shap
from yellowbrick.regressor import PredictionError
from yellowbrick.contrib.wrapper import wrap

import datetime
from typing import Any, List, Tuple, Dict

import collections.abc as coll
import tempfile
from contextlib import contextmanager
import os

# COMMAND ----------

def get_data_to_predict(range_plan_cd: int, wshop_cd: str):
    return spark.sql(
        f'''SELECT DISTINCT
        clients.client_id,
        items.wshop_cd,
        items.mdm_item_sid,
        ldlitems.mdm_item_name,
        CAST(items.item_ian AS INTEGER),
        CAST(items.range_plan_cd AS INTEGER),
        items.src_familygrp_cd,
        items.brand_sid,
        CASE(CASE WHEN items.brand_type_cd = 'EM' THEN 1 ELSE 0 END AS STRING) AS brand_type_cd,
        DECODE(am_items.sale_promo_type_cd,  'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS promotion_medium_type, 
        items.tv_fg,
        items.sell_off_horizon,
        items.sales_price_wt AS sales_price,
        items.after_promo_qty,
        items.promo_qty,
        items.sell_off_period_end
    FROM paf.am_items AS items
    JOIN ldl.items AS ldlitems
        ON ldlitems.item_ian = items.item_ian
            AND ldlitems.range_plan_cd = items.range_plan_cd
    JOIN ldl.clients AS clients
        ON clients.client_cd = items.wshop_cd
    WHERE items.range_plan_cd BETWEEN {range_plan_cd[0]} and {range_plan_cd[-1]}
        AND items.wshop_cd = "{wshop_cd}"'''
    )

# COMMAND ----------

def get_baseline_wt(wshop_cd: Tuple[str]):
    return (
        spark.sql(
            f"""
    SELECT
        *
    FROM (
        SELECT
            items.item_ian,
            items.wshop_cd,
            items.range_plan_cd,
            CASE WHEN preds.item_ian IS NOT NULL THEN 1 ELSE 0 END AS has_predecessors,
            pred1.pre_item_ian AS pred1_item_ian,
            pred2.pre_item_ian AS pred2_item_ian,
            COALESCE(
                pred1_targets.target_sales,
                pred2_targets.target_sales, 
                fambrands_stats.wt_avg_fambrand_sum_sales_first_wt, 
                familygrp_stats.wt_avg_familygrp_sum_sales_first_wt
            ) AS baseline_wt
        FROM paf.am_items AS items
        
        LEFT JOIN ( 
            SELECT DISTINCT
                wshop_cd,
                item_ian
            FROM paf.predecessors_filtered
        ) AS preds
            ON preds.item_ian = items.item_ian
                AND preds.wshop_cd = items.wshop_cd
        
        LEFT JOIN paf.predecessors_filtered AS pred1
            ON pred1.item_ian = items.item_ian
                AND pred1.wshop_cd = items.wshop_cd
                AND pred1.predecessor_x = 1
        LEFT JOIN paf.target_wt AS pred1_targets
            ON pred1_targets.item_ian = pred1.pre_item_ian
                AND pred1_targets.wshop_cd = pred1.wshop_cd
                AND DATE_ADD(pred1_targets.promo_sale_start_dt, CAST(pred1_targets.sum_from_n_days AS INTEGER)) <= items.cutoff_time
        
        LEFT JOIN paf.predecessors_filtered AS pred2
            ON pred2.item_ian = items.item_ian
                AND pred2.wshop_cd = items.wshop_cd
                AND pred2.predecessor_x = 2
        LEFT JOIN paf.target_wt AS pred2_targets
            ON pred2_targets.item_ian = pred2.pre_item_ian
                AND pred2_targets.wshop_cd = pred2.wshop_cd
                AND DATE_ADD(pred2_targets.promo_sale_start_dt, CAST(pred2_targets.sum_from_n_days AS INTEGER)) <= items.cutoff_time
        
        LEFT JOIN paf_feature_store.wt_sales_stats_of_familygrp_brand_combination_at_selection AS fambrands_stats
            ON fambrands_stats.src_familygrp_cd = items.src_familygrp_cd
                AND fambrands_stats.wshop_cd = items.wshop_cd
                AND fambrands_stats.brand_sid = items.brand_sid
                AND fambrands_stats.range_plan_cd = items.range_plan_cd
        
        LEFT JOIN paf_feature_store.wt_sales_stats_of_familygrp_at_selection AS familygrp_stats
            ON familygrp_stats.src_familygrp_cd = items.src_familygrp_cd
                AND familygrp_stats.wshop_cd = items.wshop_cd
                AND familygrp_stats.range_plan_cd = items.range_plan_cd
                
    WHERE items.wshop_cd  IN {wshop_cd}
    )
    WHERE baseline_wt IS NOT NULL
        
"""
        )
        .toPandas()
        .drop_duplicates()
    )

# COMMAND ----------

def get_baseline_nl( wshop_cd: Tuple[str]):
    return spark.sql(
        f"""
    SELECT
        *
    FROM (
        SELECT
            items.wshop_cd,
            items.item_ian,
            items.range_plan_cd,
            CASE WHEN preds.item_ian IS NOT NULL THEN 1 ELSE 0 END AS has_predecessors,
            baseline_nl.pre_item_ian AS pred_item_ian,
            COALESCE(
                baseline_nl.mean_sales,
                fambrands_stats.nl_avg_fambrand_weekly_sales, 
                familygrp_stats.nl_avg_familygrp_weekly_sales
            ) AS baseline_nl
        FROM paf.am_items AS items
                
        LEFT JOIN (
            SELECT DISTINCT
                item_ian
            FROM paf.predecessors_filtered
        ) AS preds
            ON preds.item_ian = items.item_ian
        
        LEFT JOIN paf.baseline_nl AS baseline_nl
            ON baseline_nl.item_suc = items.item_ian
                AND baseline_nl.wshop_cd = items.wshop_cd

        LEFT JOIN paf_feature_store.nl_sales_stats_of_familygrp_brand_combination_at_selection AS fambrands_stats
            ON fambrands_stats.src_familygrp_cd = items.src_familygrp_cd
                AND fambrands_stats.brand_sid = items.brand_sid
                AND fambrands_stats.range_plan_cd = items.range_plan_cd
                AND fambrands_stats.wshop_cd = items.wshop_cd
        
        LEFT JOIN paf_feature_store.nl_sales_stats_of_familygrp_at_selection AS familygrp_stats
            ON familygrp_stats.src_familygrp_cd = items.src_familygrp_cd
                AND familygrp_stats.range_plan_cd = items.range_plan_cd
                AND familygrp_stats.wshop_cd = items.wshop_cd
        

        WHERE items.wshop_cd  IN {wshop_cd}
    )
    WHERE baseline_nl IS NOT NULL
"""
    ).toPandas()

# COMMAND ----------

class DataProcessor:
    def __init__(self, wshop_cd = None, coal_logic: bool = True):
        self.fs = feature_store.FeatureStoreClient()
        self.wshop_cd = wshop_cd
        self.mapping = {'HHZ': 1, 'OHZ': 2, 'EU': 3, 'ED': 4, 'BM': 5}
        self.predict_set = False
        self.coal_logic = coal_logic
        

    def load_preprocessed_data(self):
        self.predict_set = False
        self._load_base_data()
        self._load_feature_store()
        self._preprocessing_sql() if self.coal_logic else self._preprocessing_sql_explode()
        
        return self.preprocessed_data_sql if self.coal_logic else self.preprocessed_data_exploded
        #self._preprocessing_pandas()
        
        
    def load_preprocessed_predict_data(self, range_plan_cd:List):
        self.predict_set = True
        
        self._load_predict_set(range_plan_cd=range_plan_cd)
        self._load_feature_store()
        self._preprocessing_sql() if self.coal_logic else self._preprocessing_sql_explode()
        
        return self.preprocessed_data_sql if self.coal_logic else self.preprocessed_data_exploded
        
        
    def _load_predict_set(self, range_plan_cd: List) -> pd.DataFrame:
        """
        Load base df of current AMs. This df is missing variables like promotion_medium_type. 
        These need to be set by end-user in frontend
        """
        self.predict_df =  spark.sql(f"""
                SELECT DISTINCT
                    CAST(RIGHT(clients.client_id, 2) AS INT) AS client_id,
                    RIGHT(clients.client_cd, 2) AS wshop_cd,
                    items.mdm_item_sid,
                    items.item_ian,
                    items.range_plan_cd,
                    CAST(CASE WHEN brands.brand_type_cd = 'EM' THEN 1 ELSE 0 END AS STRING) AS brand_type_cd,
                    src_familygrp_cd,
                    items.brand_sid,
                    CONCAT(src_familygrp_cd, '_', items.brand_sid) AS brand_famgrp
                FROM ldl.items_orderable_by_clients AS orderable
                JOIN ldl.items AS items
                    ON orderable.mdm_item_sid = items.mdm_item_sid
                JOIN ldl.brands AS brands
                    ON brands.brand_sid = items.brand_sid
                JOIN ldl.clients AS clients
                    ON orderable.client_id = clients.client_id
                        AND clients.client_id IN (1022, 1023, 1002, 1006, 1024, 1029)
                        AND items.range_plan_cd BETWEEN {range_plan_cd[0]} and {range_plan_cd[-1]}
                        AND RIGHT(clients.client_cd, 2) = "{self.wshop_cd}"
                    """)

    def _load_base_data(self):
        """
        Loads base dataframe to train WT & NL models.
        Features from feature_store need to be joined to this df.
        """
        
        self.base_df = spark.sql(
            f"""
                WITH nl_prices AS (
                    SELECT
                        wshop_cd,
                        item_ian,
                        MODE(continuous_sales_price) AS nl_price
                    FROM paf.sales
                    WHERE sales_type = 'nl'
                    GROUP BY 1,2
                       )
                       
                SELECT DISTINCT 
                    am_items.wshop_cd,
                    clients.client_id,
                    CAST(am_items.mdm_item_sid AS INT),
                    CAST(am_items.item_ian AS INT),
                    CAST(am_items.range_plan_cd AS INT),
                    am_items.src_familygrp_cd,
                    CAST(am_items.brand_sid AS INT),
                    CAST(CASE WHEN am_items.brand_type_cd = 'EM' THEN 1 ELSE 0 END AS STRING) AS brand_type_cd,
                    CAST(DECODE(am_items.sale_promo_type_cd, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS INT) AS promotion_medium_type,
                    CAST(tv_fg AS INT) AS tv_fg, 
                    CAST(am_items.sell_off_horizon AS INT),
                    CAST(am_items.wt_price AS FLOAT), -- AS sales_price,
                    CAST(nl_price AS FLOAT),
                    CAST(am_items.promo_qty AS INT),
                    CAST(am_items.after_promo_qty AS INT),
                    CAST(targets_wt.target_sales AS FLOAT) AS target_wt,
                    CAST(targets_nl.target_sales AS FLOAT) AS target_nl
                FROM paf.am_items AS am_items
                LEFT JOIN paf.target_wt AS targets_wt
                    ON targets_wt.wshop_cd = am_items.wshop_cd
                        AND targets_wt.item_ian = am_items.item_ian
                LEFT JOIN paf.target_nl AS targets_nl
                    ON targets_nl.wshop_cd = am_items.wshop_cd
                        AND targets_nl.item_ian = am_items.item_ian
                JOIN ldl.clients AS clients
                    ON clients.client_cd = am_items.wshop_cd
                JOIN nl_prices AS nl_price
                    ON nl_price.wshop_cd = am_items.wshop_cd
                        AND nl_price.item_ian = am_items.item_ian
                WHERE (targets_wt.target_sales > 0 OR targets_nl.target_sales > 0)
                    AND am_items.wshop_cd IN (
                            SELECT 
                            CASE 
                                WHEN "{self.wshop_cd}" != "None" THEN  "{self.wshop_cd}"
                                WHEN "{self.wshop_cd}" == "None" THEN wshop_cd
                            END AS wshop_selected
                            FROM paf.am_items
                        )
                    AND CAST(DECODE(am_items.sale_promo_type_cd, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS INT) IS NOT NULL
                ORDER BY range_plan_cd
            """
        )

    def _load_feature_store(self, predict_set:bool = False):
        """
        Raw features get pulled from feature_store and joined to base df on lookup-keys.
        """
        self.base_df.createOrReplaceTempView("base")
        self.feature_df = spark.sql(
            """
            SELECT 
            base.wshop_cd,
            base.item_ian,
            base.range_plan_cd,
            wt_price,
            nl_price,
            target_wt,
            target_nl,
            sell_off_horizon,
            promotion_medium_type,
            brand_type_cd,
            promo_qty,
            after_promo_qty,
            tv_fg,
            wt_avg_pred_sale_promo_type,
                wt_avg_pred_sum_sales_first_wt, 
                wt_decay_25_avg_pred_sum_sales_first_wt, 
                wt_decay_50_avg_pred_sum_sales_first_wt, 
                wt_decay_75_avg_pred_sum_sales_first_wt, 
                wt_pred_items_used_in_avg, 
                wt_avg_pred_wt_price, 
                decay_25_wt_avg_pred_wt_price, 
                decay_50_wt_avg_pred_wt_price, 
                decay_75_wt_avg_pred_wt_price, 
                wt_single_pred_sum_sales_first_wt, 
                wt_single_pred_item_ian, 
                wt_single_pred_sale_promo_type, 
                wt_pred_num_promo, 
                wt_pred_num_promo_all, 
                wt_pred_promo_date,
            wt_avg_sim_sale_promo_type, 
                wt_avg_sim_sum_sales_first_wt, 
                decay_25_avg_sim_sum_sales_first_wt, 
                decay_50_avg_sim_sum_sales_first_wt, 
                decay_75_avg_sim_sum_sales_first_wt,
                wt_sim_items_used_in_avg, 
                wt_avg_sim_wt_price,
                decay_25_wt_avg_sim_wt_price, 
                decay_50_wt_avg_sim_wt_price, 
                decay_75_wt_avg_sim_wt_price, 
                wt_single_sim_sum_sales_first_wt, 
                wt_single_sim_item_ian, 
                wt_single_sim_sale_promo_type, 
                wt_sim_num_promo, 
                wt_sim_num_promo_all, 
                wt_sim_promo_date,
            wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_25_wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_50_wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_75_wt_avg_fambrand_sum_sales_first_wt_hhz,
                decay_wt_avg_fambrand_sum_sales_first_wt, 
                wt_avg_fambrand_based_on, 
                wt_avg_fambrand_wt_price_hhz, 
                decay_wt_avg_fambrand_wt_price_hhz,
            wt_avg_familygrp_sum_sales_first_wt_hhz, 
                decay_25_wt_avg_familygrp_sum_sales_first_wt_hhz,
                decay_50_wt_avg_familygrp_sum_sales_first_wt_hhz, 
                decay_75_wt_avg_familygrp_sum_sales_first_wt_hhz,
                decay_wt_avg_familygrp_sum_sales_first_wt, 
                wt_avg_familygrp_based_on, 
                wt_avg_familygrp_wt_price_hhz,
            nl_single_pred_avg_weekly_sales, 
                nl_single_pred_item_ian, 
                nl_avg_preds_sales_price, 
                decay_25_nl_avg_preds_sales_price, 
                decay_50_nl_avg_preds_sales_price, 
                decay_75_nl_avg_preds_sales_price, 
                nl_pred_items_used_in_avg, 
                nl_avg_preds_weekly_sales, 
                decay_25_nl_avg_preds_weekly_sales, 
                decay_50_nl_avg_preds_weekly_sales, 
                decay_75_nl_avg_preds_weekly_sales,
            nl_single_sim_avg_weekly_sales,
                nl_single_sim_item_ian, 
                nl_avg_sim_weekly_sales, 
                nl_sim_items_used_in_avg, 
                nl_avg_sim_sales_price, 
                decay_25_nl_avg_sim_sales_price, 
                decay_50_nl_avg_sim_sales_price, 
                decay_75_nl_avg_sim_sales_price, 
                decay_25_nl_avg_sim_weekly_sales, 
                decay_50_nl_avg_sim_weekly_sales,
                decay_75_nl_avg_sim_weekly_sales,
            nl_avg_fambrand_weekly_sales, 
                decay_25_nl_avg_fambrand_weekly_sales,
                decay_50_nl_avg_fambrand_weekly_sales, 
                decay_75_nl_avg_fambrand_weekly_sales, 
                nl_avg_fambrand_sales_price, 
                decay_25_nl_avg_fambrand_sales_price,
                decay_50_nl_avg_fambrand_sales_price, 
                decay_75_nl_avg_fambrand_sales_price,
            nl_avg_familygrp_weekly_sales,
                decay_25_nl_avg_familygrp_weekly_sales, 
                decay_50_nl_avg_familygrp_weekly_sales, 
                decay_75_nl_avg_familygrp_weekly_sales, 
                nl_avg_familygrp_sales_price, 
                decay_25_nl_avg_familygrp_sales_price, 
                decay_50_nl_avg_familygrp_sales_price, 
                decay_75_nl_avg_familygrp_sales_price,
            avg_fambrand_d7_qty_sales_sum,
            avg_preds_d7_qty_sales_sum,
            avg_preds_daily_sales_nl, 
                decay_25_avg_preds_daily_sales_nl, 
                decay_50_avg_preds_daily_sales_nl, 
                decay_75_avg_preds_daily_sales_nl, 
                trend_preds_daily_sales_nl, 
                trend_preds_daily_sales, 
                avg_preds_daily_sales,
                decay_25_avg_preds_daily_sales,
                decay_50_avg_preds_daily_sales, 
                decay_75_avg_preds_daily_sales, 
                std_preds_daily_sales, 
                std_preds_daily_sales_nl,
            avg_familygrp_daily_sales_nl, 
                decay_25_avg_familygrp_daily_sales_nl, 
                decay_50_avg_familygrp_daily_sales_nl, 
                decay_75_avg_familygrp_daily_sales_nl, 
                avg_familygrp_daily_sales, 
                decay_25_avg_familygrp_daily_sales, 
                decay_50_avg_familygrp_daily_sales, 
                decay_75_avg_familygrp_daily_sales, 
                trend_familygrp_daily_sales,
                std_familygrp_daily_sales, 
                std_familygrp_daily_sales_nl,
            avg_fambrands_daily_sales_nl, 
                decay_25_avg_fambrands_daily_sales_nl, 
                decay_50_avg_fambrands_daily_sales_nl, 
                decay_75_avg_fambrands_daily_sales_nl,
                avg_fambrands_daily_sales,
                decay_25_avg_fambrands_daily_sales,
                decay_50_avg_fambrands_daily_sales, 
                decay_75_avg_fambrands_daily_sales, 
                trend_fambrands_daily_sales, 
                std_fambrands_daily_sales, 
                std_fambrands_daily_sales_nl,
            avg_sim_daily_sales, 
                decay_25_avg_sim_daily_sales, 
                decay_50_avg_sim_daily_sales,
                decay_75_avg_sim_daily_sales, 
                avg_sim_daily_sales_nl, 
                decay_25_avg_sim_daily_sales_nl, 
                decay_50_avg_sim_daily_sales_nl, 
                decay_75_avg_sim_daily_sales_nl, 
                std_sim_daily_sales, 
                std_sim_daily_sales_nl,
                range_plan_month,
            item_name_embeddings_0,
            item_name_embeddings_1,
            item_name_embeddings_2
            FROM base
            LEFT JOIN (
                SELECT
                wt_avg_pred_sale_promo_type,
                wt_avg_pred_sum_sales_first_wt, 
                wt_decay_25_avg_pred_sum_sales_first_wt, 
                wt_decay_50_avg_pred_sum_sales_first_wt, 
                wt_decay_75_avg_pred_sum_sales_first_wt, 
                wt_pred_items_used_in_avg, 
                wt_avg_pred_wt_price, 
                decay_25_wt_avg_pred_wt_price, 
                decay_50_wt_avg_pred_wt_price, 
                decay_75_wt_avg_pred_wt_price, 
                wt_single_pred_sum_sales_first_wt, 
                wt_single_pred_item_ian, 
                wt_single_pred_sale_promo_type, 
                wt_pred_num_promo, 
                wt_pred_num_promo_all, 
                wt_pred_promo_date,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.wt_sales_stats_of_predecessors_at_successor_selection 
            ) AS wt_pred
                ON base.wshop_cd = wt_pred.wshop_cd
                AND base.item_ian = wt_pred.item_ian
                AND base.range_plan_cd = wt_pred.range_plan_cd
            LEFT JOIN (
                SELECT 
                wt_avg_sim_sale_promo_type, 
                wt_avg_sim_sum_sales_first_wt, 
                decay_25_avg_sim_sum_sales_first_wt, 
                decay_50_avg_sim_sum_sales_first_wt, 
                decay_75_avg_sim_sum_sales_first_wt,
                wt_sim_items_used_in_avg, 
                wt_avg_sim_wt_price,
                decay_25_wt_avg_sim_wt_price, 
                decay_50_wt_avg_sim_wt_price, 
                decay_75_wt_avg_sim_wt_price, 
                wt_single_sim_sum_sales_first_wt, 
                wt_single_sim_item_ian, 
                wt_single_sim_sale_promo_type, 
                wt_sim_num_promo, 
                wt_sim_num_promo_all, 
                wt_sim_promo_date,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.wt_sales_stats_of_similar_at_successor_selection 
            ) AS wt_similar
                ON base.wshop_cd = wt_similar.wshop_cd
                AND base.item_ian = wt_similar.item_ian
                AND base.range_plan_cd = wt_similar.range_plan_cd
            LEFT JOIN(
                SELECT 
                wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_25_wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_50_wt_avg_fambrand_sum_sales_first_wt_hhz, 
                decay_75_wt_avg_fambrand_sum_sales_first_wt_hhz,
                decay_wt_avg_fambrand_sum_sales_first_wt, 
                wt_avg_fambrand_based_on, 
                wt_avg_fambrand_wt_price_hhz, 
                decay_wt_avg_fambrand_wt_price_hhz,
                wshop_cd,
                src_familygrp_cd,
                brand_sid,
                range_plan_cd
                FROM paf_feature_store.wt_sales_stats_of_familygrp_brand_combination_at_selection 
            ) AS wt_fambrd
                ON base.wshop_cd = wt_fambrd.wshop_cd
                AND base.src_familygrp_cd = wt_fambrd.src_familygrp_cd
                AND base.brand_sid = wt_fambrd.brand_sid
                AND base.range_plan_cd = wt_fambrd.range_plan_cd
            LEFT JOIN(
                SELECT 
                wt_avg_familygrp_sum_sales_first_wt_hhz, 
                decay_25_wt_avg_familygrp_sum_sales_first_wt_hhz,
                decay_50_wt_avg_familygrp_sum_sales_first_wt_hhz, 
                decay_75_wt_avg_familygrp_sum_sales_first_wt_hhz,
                decay_wt_avg_familygrp_sum_sales_first_wt, 
                wt_avg_familygrp_based_on, 
                wt_avg_familygrp_wt_price_hhz,
                wshop_cd,
                src_familygrp_cd,
                range_plan_cd
                FROM paf_feature_store.wt_sales_stats_of_familygrp_at_selection 
            ) AS wt_familygrp
                ON base.wshop_cd = wt_familygrp.wshop_cd
                AND base.src_familygrp_cd = wt_familygrp.src_familygrp_cd
                AND base.range_plan_cd = wt_familygrp.range_plan_cd
            LEFT JOIN(
                SELECT
                nl_single_pred_avg_weekly_sales, 
                nl_single_pred_item_ian, 
                nl_avg_preds_sales_price, 
                decay_25_nl_avg_preds_sales_price, 
                decay_50_nl_avg_preds_sales_price, 
                decay_75_nl_avg_preds_sales_price, 
                nl_pred_items_used_in_avg, 
                nl_avg_preds_weekly_sales, 
                decay_25_nl_avg_preds_weekly_sales, 
                decay_50_nl_avg_preds_weekly_sales, 
                decay_75_nl_avg_preds_weekly_sales,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.nl_sales_stats_of_predecessors_at_successor_selection 
            ) AS nl_pred
                ON base.wshop_cd = nl_pred.wshop_cd
                AND base.item_ian = nl_pred.item_ian
                AND base.range_plan_cd = nl_pred.range_plan_cd
            LEFT JOIN(
                SELECT
                nl_single_sim_avg_weekly_sales,
                nl_single_sim_item_ian, 
                nl_avg_sim_weekly_sales, 
                nl_sim_items_used_in_avg, 
                nl_avg_sim_sales_price, 
                decay_25_nl_avg_sim_sales_price, 
                decay_50_nl_avg_sim_sales_price, 
                decay_75_nl_avg_sim_sales_price, 
                decay_25_nl_avg_sim_weekly_sales, 
                decay_50_nl_avg_sim_weekly_sales,
                decay_75_nl_avg_sim_weekly_sales,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.nl_sales_stats_of_similar_at_successor_selection 
            ) AS nl_similar
                ON base.wshop_cd = nl_similar.wshop_cd
                AND base.item_ian = nl_similar.item_ian
                AND base.range_plan_cd = nl_similar.range_plan_cd
            LEFT JOIN(
                SELECT
                nl_avg_fambrand_weekly_sales, 
                decay_25_nl_avg_fambrand_weekly_sales,
                decay_50_nl_avg_fambrand_weekly_sales, 
                decay_75_nl_avg_fambrand_weekly_sales, 
                nl_avg_fambrand_sales_price, 
                decay_25_nl_avg_fambrand_sales_price,
                decay_50_nl_avg_fambrand_sales_price, 
                decay_75_nl_avg_fambrand_sales_price,
                wshop_cd,
                src_familygrp_cd,
                brand_sid,
                range_plan_cd
                FROM paf_feature_store.nl_sales_stats_of_familygrp_brand_combination_at_selection 
            ) AS nl_fambrand
                ON base.wshop_cd = nl_fambrand.wshop_cd
                AND base.src_familygrp_cd = nl_fambrand.src_familygrp_cd
                AND base.brand_sid = nl_fambrand.brand_sid
                AND base.range_plan_cd = nl_fambrand.range_plan_cd
            LEFT JOIN(
                SELECT
                nl_avg_familygrp_weekly_sales,
                decay_25_nl_avg_familygrp_weekly_sales, 
                decay_50_nl_avg_familygrp_weekly_sales, 
                decay_75_nl_avg_familygrp_weekly_sales, 
                nl_avg_familygrp_sales_price, 
                decay_25_nl_avg_familygrp_sales_price, 
                decay_50_nl_avg_familygrp_sales_price, 
                decay_75_nl_avg_familygrp_sales_price,
                wshop_cd,
                src_familygrp_cd,
                range_plan_cd
                FROM paf_feature_store.nl_sales_stats_of_familygrp_at_selection 
            ) AS nl_famgrp
                ON base.wshop_cd = nl_famgrp.wshop_cd
                AND base.src_familygrp_cd = nl_famgrp.src_familygrp_cd
                AND base.range_plan_cd = nl_famgrp.range_plan_cd
            LEFT JOIN(
                SELECT
                avg_fambrand_d7_qty_sales_sum,
                client_id,
                src_familygrp_cd,
                brand_sid,
                range_plan_cd
                FROM nfdf_feature_store.sales_stats_of_familygrp_brand_combination_at_selection 
            ) AS store_famgrp
                ON base.client_id = store_famgrp.client_id
                AND base.src_familygrp_cd = store_famgrp.src_familygrp_cd
                AND base.brand_sid = store_famgrp.brand_sid
                AND base.range_plan_cd = store_famgrp.range_plan_cd
            LEFT JOIN(
                SELECT
                avg_preds_d7_qty_sales_sum,
                client_id,
                mdm_item_sid
                FROM nfdf_feature_store.sales_stats_of_predecessors_at_successor_selection 
            ) AS store_preds
                ON base.client_id = store_preds.client_id
                AND base.mdm_item_sid = store_preds.mdm_item_sid
            LEFT JOIN(
                SELECT
                avg_preds_daily_sales_nl, 
                decay_25_avg_preds_daily_sales_nl, 
                decay_50_avg_preds_daily_sales_nl, 
                decay_75_avg_preds_daily_sales_nl, 
                trend_preds_daily_sales_nl, 
                trend_preds_daily_sales, 
                avg_preds_daily_sales,
                decay_25_avg_preds_daily_sales,
                decay_50_avg_preds_daily_sales, 
                decay_75_avg_preds_daily_sales, 
                std_preds_daily_sales, 
                std_preds_daily_sales_nl,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.sales_stats_of_predecessors_at_successor_selection 
            ) AS stats_preds
                ON base.wshop_cd = stats_preds.wshop_cd
                AND base.item_ian = stats_preds.item_ian
                AND base.range_plan_cd = stats_preds.range_plan_cd
            LEFT JOIN(
                SELECT
                avg_familygrp_daily_sales_nl, 
                decay_25_avg_familygrp_daily_sales_nl, 
                decay_50_avg_familygrp_daily_sales_nl, 
                decay_75_avg_familygrp_daily_sales_nl, 
                avg_familygrp_daily_sales, 
                decay_25_avg_familygrp_daily_sales, 
                decay_50_avg_familygrp_daily_sales, 
                decay_75_avg_familygrp_daily_sales, 
                trend_familygrp_daily_sales,
                std_familygrp_daily_sales, 
                std_familygrp_daily_sales_nl,
                wshop_cd,
                src_familygrp_cd,
                range_plan_cd
                FROM paf_feature_store.sales_stats_of_familygrp_at_selection 
            ) AS stats_famgrp
                ON base.wshop_cd = stats_famgrp.wshop_cd
                AND base.src_familygrp_cd = stats_famgrp.src_familygrp_cd
                AND base.range_plan_cd = stats_famgrp.range_plan_cd
            LEFT JOIN(
                SELECT
                avg_fambrands_daily_sales_nl, 
                decay_25_avg_fambrands_daily_sales_nl, 
                decay_50_avg_fambrands_daily_sales_nl, 
                decay_75_avg_fambrands_daily_sales_nl,
                avg_fambrands_daily_sales,
                decay_25_avg_fambrands_daily_sales,
                decay_50_avg_fambrands_daily_sales, 
                decay_75_avg_fambrands_daily_sales, 
                trend_fambrands_daily_sales, 
                std_fambrands_daily_sales, 
                std_fambrands_daily_sales_nl,
                wshop_cd,
                src_familygrp_cd,
                brand_sid,
                range_plan_cd
                FROM paf_feature_store.sales_stats_of_familygrp_brand_combination_at_selection 
            ) AS stats_familybrand
                ON base.wshop_cd = stats_familybrand.wshop_cd
                AND base.src_familygrp_cd = stats_familybrand.src_familygrp_cd
                AND base.brand_sid = stats_familybrand.brand_sid
                AND base.range_plan_cd = stats_familybrand.range_plan_cd
            LEFT JOIN(
                SELECT
                avg_sim_daily_sales, 
                decay_25_avg_sim_daily_sales, 
                decay_50_avg_sim_daily_sales,
                decay_75_avg_sim_daily_sales, 
                avg_sim_daily_sales_nl, 
                decay_25_avg_sim_daily_sales_nl, 
                decay_50_avg_sim_daily_sales_nl, 
                decay_75_avg_sim_daily_sales_nl, 
                std_sim_daily_sales, 
                std_sim_daily_sales_nl,
                wshop_cd,
                item_ian,
                range_plan_cd
                FROM paf_feature_store.sales_stats_similar_at_succesor_selection 
            ) AS stats_similar
                ON base.wshop_cd = stats_similar.wshop_cd
                AND base.item_ian = stats_similar.item_ian
                AND base.range_plan_cd = stats_similar.range_plan_cd
            LEFT JOIN paf_feature_store.semantic_item_name_pca_embeddings AS embeddings
                ON base.mdm_item_sid = embeddings.mdm_item_sid
            LEFT JOIN(
                SELECT
                range_plan_month,
                range_plan_cd
                FROM nfdf_feature_store.range_plan_features 
            ) AS range_plans
                ON base.range_plan_cd = range_plans.range_plan_cd
            """
        )

       

    def _preprocessing_sql(self):
        """
        Build final features for NL & WT Models.
        """
        # create temp view
        self.feature_df.createOrReplaceTempView("inference_data")
        self.preprocessed_data_sql = spark.sql(
            f"""
            
            SELECT
                wshop_cd,
                item_ian,
                range_plan_cd,
                brand_type_cd,
                item_name_embeddings_0,
                item_name_embeddings_1,
                item_name_embeddings_2,
                avg_wt_price,
                decay_25_avg_wt_price,
                decay_50_avg_wt_price,
                decay_75_avg_wt_price,
                avg_nl_price,
                decay_25_avg_nl_price,
                decay_50_avg_nl_price,
                decay_75_avg_nl_price,
                {'''tv_fg,
                    promotion_medium_type,
                    wt_price,
                    sell_off_horizon,
                    wt_diff_in_sales_price,
                    wt_diff_25_in_sales_price,
                    wt_diff_50_in_sales_price,
                    wt_diff_75_in_sales_price,
                    nl_diff_in_sales_price,
                    nl_diff_25_in_sales_price,
                    nl_diff_50_in_sales_price,
                    nl_diff_75_in_sales_price,
                    promo_qty,
                    after_promo_qty,
                    target_wt,
                    target_nl,''' if not self.predict_set else ''}
                daily_sales,
                daily_sales_nl,
                std_daily_sales,
                std_daily_sales_nl,
                decay_25_daily_sales,
                decay_50_daily_sales,
                decay_75_daily_sales,
                decay_25_daily_sales_nl,
                decay_50_daily_sales_nl,
                decay_75_daily_sales_nl,
                trend_daily_sales,
                nl_avg_weekly_sales,
                nl_single_weekly_sales,
                decay_25_nl_avg_weekly_sales,
                decay_50_nl_avg_weekly_sales,
                decay_75_nl_avg_weekly_sales,
                wt_avg_sum_sales,
                decay_25_wt_avg_sum_sales,
                decay_50_wt_avg_sum_sales,
                decay_75_wt_avg_sum_sales,
                wt_single_sum_sales,
                store_sales_wt,
                num_promo,
                promo_date,
                num_promo_all,
                nl_avg_based_on,
                nl_single_based_on,
                nl_items_used,
                nl_avg_sales_price_in_items_used,
                wt_avg_based_on,
                wt_single_based_on,
                wt_items_used,
                wt_avg_wt_price_in_items_used,
                wt_single_item_ian,
                nl_single_item_ian,
                wt_single_promo_type,
                wt_avg_promo_type
            FROM(
                SELECT
                    *,
                    CASE 
                        WHEN nl_avg_preds_weekly_sales IS NOT NULL THEN 1
                        WHEN nl_avg_sim_weekly_sales IS NOT NULL THEN 2
                        WHEN nl_avg_fambrand_weekly_sales IS NOT NULL THEN 3
                        WHEN nl_avg_familygrp_weekly_sales IS NOT NULL THEN 4
                        ELSE 0
                    END AS nl_avg_based_on,
                    CASE 
                        WHEN nl_single_pred_avg_weekly_sales IS NOT NULL THEN 1
                        WHEN nl_single_sim_avg_weekly_sales IS NOT NULL THEN 2
                        WHEN nl_avg_fambrand_weekly_sales IS NOT NULL THEN 3
                        WHEN nl_avg_familygrp_weekly_sales IS NOT NULL THEN 4
                        ELSE 0
                    END AS nl_single_based_on,
                    CASE
                        WHEN nl_avg_preds_weekly_sales IS NOT NULL THEN nl_pred_items_used_in_avg
                        WHEN nl_avg_sim_weekly_sales IS NOT NULL THEN nl_sim_items_used_in_avg
                        ELSE NULL
                    END AS nl_items_used,
                    
                    COALESCE(nl_avg_preds_sales_price, nl_avg_sim_sales_price) AS nl_avg_sales_price_in_items_used,
                    
                    CASE 
                        WHEN wt_avg_pred_sum_sales_first_wt IS NOT NULL THEN 1
                        WHEN wt_avg_sim_sum_sales_first_wt IS NOT NULL THEN 2
                        WHEN wt_avg_fambrand_sum_sales_first_wt_hhz IS NOT NULL THEN 3
                        WHEN wt_avg_familygrp_sum_sales_first_wt_hhz IS NOT NULL THEN 4
                        ELSE 0
                    END AS wt_avg_based_on,
                    CASE 
                        WHEN wt_single_pred_sum_sales_first_wt IS NOT NULL THEN 1
                        WHEN wt_single_sim_sum_sales_first_wt IS NOT NULL THEN 2
                        WHEN wt_avg_fambrand_sum_sales_first_wt_hhz IS NOT NULL THEN 3
                        WHEN wt_avg_familygrp_sum_sales_first_wt_hhz IS NOT NULL THEN 4
                        ELSE 0
                    END AS wt_single_based_on,
                    CASE
                        WHEN wt_avg_pred_sum_sales_first_wt IS NOT NULL THEN wt_pred_items_used_in_avg
                        WHEN wt_avg_sim_sum_sales_first_wt IS NOT NULL THEN wt_sim_items_used_in_avg
                        ELSE NULL
                    END AS wt_items_used,
                    
                    COALESCE(wt_avg_pred_wt_price, wt_avg_sim_wt_price) AS wt_avg_wt_price_in_items_used,
                    --COALESCE(wt_decay_avg_pred_wt_price, wt_decay_avg_sim_wt_price) AS wt_decay_avg_wt_price_in_items_used,
                    
                    {'''CAST(wt_price - avg_wt_price AS FLOAT) AS wt_diff_in_sales_price, CAST(nl_price - avg_nl_price AS FLOAT) AS nl_diff_in_sales_price,
                    CAST(wt_price - decay_25_avg_wt_price AS FLOAT) AS wt_diff_25_in_sales_price, CAST(nl_price - decay_25_avg_nl_price AS FLOAT) AS nl_diff_25_in_sales_price,
                    CAST(wt_price - decay_50_avg_wt_price AS FLOAT) AS wt_diff_50_in_sales_price, CAST(nl_price - decay_50_avg_nl_price AS FLOAT) AS nl_diff_50_in_sales_price,
                    CAST(wt_price - decay_75_avg_wt_price AS FLOAT) AS wt_diff_75_in_sales_price, CAST(nl_price - decay_75_avg_nl_price AS FLOAT) AS nl_diff_75_in_sales_price,''' if not self.predict_set else ''}

                    /* WT & NL SINGLE ITEM_IAN */
                    COALESCE(wt_single_pred_item_ian, wt_single_sim_item_ian) AS wt_single_item_ian,
                    COALESCE(nl_single_pred_item_ian, nl_single_sim_item_ian) AS nl_single_item_ian,

                    /* WT SINGLE & AVG SALE PROMO TYPE */
                    CAST(DECODE(COALESCE(wt_single_pred_sale_promo_type, wt_single_sim_sale_promo_type, wt_avg_fambrand_based_on, wt_avg_familygrp_based_on),
                    'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS STRING) AS wt_single_promo_type,
                    CAST(DECODE(COALESCE(wt_avg_pred_sale_promo_type, wt_avg_sim_sale_promo_type, wt_avg_fambrand_based_on, wt_avg_familygrp_based_on),
                    'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS STRING) AS wt_avg_promo_type
                FROM(
                    SELECT
                        *,

                        /* OS SALES */
                        COALESCE(avg_preds_daily_sales, avg_sim_daily_sales, avg_fambrands_daily_sales, avg_familygrp_daily_sales) AS daily_sales,
                        COALESCE(decay_25_avg_preds_daily_sales, decay_25_avg_sim_daily_sales, decay_25_avg_fambrands_daily_sales, decay_25_avg_familygrp_daily_sales) AS decay_25_daily_sales,
                        COALESCE(decay_50_avg_preds_daily_sales, decay_50_avg_sim_daily_sales, decay_50_avg_fambrands_daily_sales, decay_50_avg_familygrp_daily_sales) AS decay_50_daily_sales,
                        COALESCE(decay_75_avg_preds_daily_sales, decay_75_avg_sim_daily_sales, decay_75_avg_fambrands_daily_sales, decay_75_avg_familygrp_daily_sales) AS decay_75_daily_sales,
                        COALESCE(avg_preds_daily_sales_nl, avg_sim_daily_sales_nl, avg_fambrands_daily_sales_nl, avg_familygrp_daily_sales_nl) AS daily_sales_nl,
                        COALESCE(decay_25_avg_preds_daily_sales_nl, decay_25_avg_sim_daily_sales_nl, decay_25_avg_fambrands_daily_sales_nl, decay_25_avg_familygrp_daily_sales_nl) AS decay_25_daily_sales_nl,
                        COALESCE(decay_50_avg_preds_daily_sales_nl, decay_50_avg_sim_daily_sales_nl, decay_50_avg_fambrands_daily_sales_nl, decay_50_avg_familygrp_daily_sales_nl) AS decay_50_daily_sales_nl,
                        COALESCE(decay_75_avg_preds_daily_sales_nl, decay_75_avg_sim_daily_sales_nl, decay_75_avg_fambrands_daily_sales_nl, decay_75_avg_familygrp_daily_sales_nl) AS decay_75_daily_sales_nl,

                        /* TREND SALES */
                        COALESCE(trend_preds_daily_sales, trend_fambrands_daily_sales, trend_familygrp_daily_sales) AS trend_daily_sales,

                        /* STANDARD DEVIATIONS */
                        COALESCE(std_preds_daily_sales, std_sim_daily_sales, std_fambrands_daily_sales, std_familygrp_daily_sales) AS std_daily_sales,
                        COALESCE(std_preds_daily_sales_nl, std_sim_daily_sales_nl, std_fambrands_daily_sales_nl, std_familygrp_daily_sales_nl) AS std_daily_sales_nl,


                        /* NL SINGLE & AVG WEEKLY SALES */
                        COALESCE(decay_25_nl_avg_preds_weekly_sales, decay_25_nl_avg_sim_weekly_sales,
                            decay_25_nl_avg_fambrand_weekly_sales, decay_25_nl_avg_familygrp_weekly_sales) AS decay_25_nl_avg_weekly_sales,
                        COALESCE(decay_50_nl_avg_preds_weekly_sales, decay_50_nl_avg_sim_weekly_sales,
                            decay_50_nl_avg_fambrand_weekly_sales, decay_50_nl_avg_familygrp_weekly_sales) AS decay_50_nl_avg_weekly_sales,
                        COALESCE(decay_75_nl_avg_preds_weekly_sales, decay_75_nl_avg_sim_weekly_sales,
                            decay_75_nl_avg_fambrand_weekly_sales, decay_75_nl_avg_familygrp_weekly_sales) AS decay_75_nl_avg_weekly_sales,
                        COALESCE(nl_avg_preds_weekly_sales, nl_avg_sim_weekly_sales, nl_avg_fambrand_weekly_sales, nl_avg_familygrp_weekly_sales) AS nl_avg_weekly_sales,
                        COALESCE(nl_single_pred_avg_weekly_sales, nl_single_sim_avg_weekly_sales, nl_avg_fambrand_weekly_sales, nl_avg_familygrp_weekly_sales) AS nl_single_weekly_sales,

                        /* WT SINGLE & AVG SUM SALES */
                        COALESCE(wt_decay_25_avg_pred_sum_sales_first_wt, decay_25_avg_sim_sum_sales_first_wt,
                            decay_25_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_25_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_25_wt_avg_sum_sales,
                        COALESCE(wt_decay_50_avg_pred_sum_sales_first_wt, decay_50_avg_sim_sum_sales_first_wt,
                            decay_50_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_50_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_50_wt_avg_sum_sales,
                        COALESCE(wt_decay_75_avg_pred_sum_sales_first_wt, decay_75_avg_sim_sum_sales_first_wt,
                            decay_75_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_75_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_75_wt_avg_sum_sales,
                        COALESCE(wt_avg_pred_sum_sales_first_wt, wt_avg_sim_sum_sales_first_wt, wt_avg_fambrand_sum_sales_first_wt_hhz, wt_avg_familygrp_sum_sales_first_wt_hhz) AS wt_avg_sum_sales,
                        COALESCE(wt_single_pred_sum_sales_first_wt, wt_single_sim_sum_sales_first_wt, wt_avg_fambrand_sum_sales_first_wt_hhz, wt_avg_familygrp_sum_sales_first_wt_hhz) AS wt_single_sum_sales,

                        /* NUMBER OF PROMOTIONS */
                        CAST(COALESCE(wt_pred_num_promo, wt_sim_num_promo) AS INT) AS num_promo,
                        CAST(COALESCE(wt_pred_num_promo_all, wt_sim_num_promo_all) AS INT) AS num_promo_all,

                        /* PROMOTION DATE */
                        COALESCE(wt_pred_promo_date, wt_sim_promo_date) AS promo_date,

                        /* STORE SALES */ 
                        COALESCE(avg_preds_d7_qty_sales_sum, avg_fambrand_d7_qty_sales_sum) AS store_sales_wt,

                        /* PRICE */
                        COALESCE(nl_avg_preds_sales_price, nl_avg_sim_sales_price, nl_avg_fambrand_sales_price, nl_avg_familygrp_sales_price) AS avg_nl_price,
                        COALESCE(decay_25_nl_avg_preds_sales_price, decay_25_nl_avg_sim_sales_price, decay_25_nl_avg_fambrand_sales_price, decay_25_nl_avg_familygrp_sales_price) AS decay_25_avg_nl_price,
                        COALESCE(decay_50_nl_avg_preds_sales_price, decay_50_nl_avg_sim_sales_price, decay_50_nl_avg_fambrand_sales_price, decay_50_nl_avg_familygrp_sales_price) AS decay_50_avg_nl_price,
                        COALESCE(decay_75_nl_avg_preds_sales_price, decay_75_nl_avg_sim_sales_price, decay_75_nl_avg_fambrand_sales_price, decay_75_nl_avg_familygrp_sales_price) AS decay_75_avg_nl_price,
                        COALESCE(wt_avg_pred_wt_price, wt_avg_sim_wt_price) AS avg_wt_price,
                        COALESCE(decay_25_wt_avg_pred_wt_price, decay_25_wt_avg_sim_wt_price) AS decay_25_avg_wt_price,
                        COALESCE(decay_50_wt_avg_pred_wt_price, decay_50_wt_avg_sim_wt_price) AS decay_50_avg_wt_price,
                        COALESCE(decay_75_wt_avg_pred_wt_price, decay_75_wt_avg_sim_wt_price) AS decay_75_avg_wt_price
                        
                    FROM inference_data
                    )
                )
            """
        )

    @udf(returnType=ArrayType(FloatType()))
    def diff_array(ar_1, ar_2):
        return [a - b for a, b in zip(ar_1, ar_2)]
    spark.udf.register("DIFF_ARRAY", diff_array)

    def _preprocessing_sql_explode(self):
            """
            Build final exploded features for NL & WT Models.
            """
            # create temp view
            self.feature_df.load_df().createOrReplaceTempView("inference_data")
            self.preprocessed_data_exploded = spark.sql(
            f"""      
            SELECT
                wshop_cd,
                item_ian,
                range_plan_cd,
                brand_type_cd,
                item_name_embeddings_0,
                item_name_embeddings_1,
                item_name_embeddings_2,
                col_combined["avg_wt_price"] AS avg_wt_price,
                col_combined["decay_25_avg_wt_price"] AS decay_25_avg_wt_price,
                col_combined["decay_50_avg_wt_price"] AS decay_50_avg_wt_price,
                col_combined["decay_75_avg_wt_price"] AS decay_75_avg_wt_price,
                col_combined["avg_nl_price"] AS avg_nl_price,
                col_combined["decay_25_avg_nl_price"] AS decay_25_avg_nl_price,
                col_combined["decay_50_avg_nl_price"] AS decay_50_avg_nl_price,
                col_combined["decay_75_avg_nl_price"] AS decay_75_avg_nl_price,
                {'''tv_fg,
                    promotion_medium_type,
                    wt_price,
                    nl_price,
                    sell_off_horizon,
                    CASE WHEN col_combined["avg_wt_price"] IS NOT NULL THEN wt_price - col_combined["avg_wt_price"] END AS wt_diff_in_sales_price,
                    CASE WHEN col_combined["decay_25_avg_wt_price"] IS NOT NULL THEN wt_price - col_combined["decay_25_avg_wt_price"] END AS wt_diff_25_in_sales_price,
                    CASE WHEN col_combined["decay_50_avg_wt_price"] IS NOT NULL THEN wt_price - col_combined["decay_50_avg_wt_price"] END AS wt_diff_50_in_sales_price,
                    CASE WHEN col_combined["decay_75_avg_wt_price"] IS NOT NULL THEN wt_price - col_combined["decay_75_avg_wt_price"] END AS wt_diff_75_in_sales_price,

                    CASE WHEN col_combined["avg_nl_price"] IS NOT NULL THEN nl_price - col_combined["avg_nl_price"] END AS nl_diff_in_sales_price,
                    CASE WHEN col_combined["decay_25_avg_nl_price"] IS NOT NULL THEN nl_price - col_combined["decay_25_avg_nl_price"] END AS nl_diff_25_in_sales_price,
                    CASE WHEN col_combined["decay_50_avg_nl_price"] IS NOT NULL THEN nl_price - col_combined["decay_50_avg_nl_price"] END AS nl_diff_50_in_sales_price,
                    CASE WHEN col_combined["decay_75_avg_nl_price"] IS NOT NULL THEN nl_price - col_combined["decay_75_avg_nl_price"] END AS nl_diff_75_in_sales_price,
                    promo_qty,
                    after_promo_qty,
                    target_wt,
                    target_nl,''' if not self.predict_set else ''}
                col_combined["daily_sales"] AS daily_sales,
                col_combined["daily_sales_nl"] AS daily_sales_nl,
                col_combined["std_daily_sales"] AS std_daily_sales,
                col_combined["std_daily_sales_nl"] AS std_daily_sales_nl,
                col_combined["decay_25_daily_sales"] AS decay_25_daily_sales,
                col_combined["decay_50_daily_sales"] AS decay_50_daily_sales,
                col_combined["decay_75_daily_sales"] AS decay_75_daily_sales,
                col_combined["decay_25_daily_sales_nl"] AS decay_25_daily_sales_nl,
                col_combined["decay_50_daily_sales_nl"] AS decay_50_daily_sales_nl,
                col_combined["decay_75_daily_sales_nl"] AS decay_75_daily_sales_nl,
                col_combined["trend_daily_sales"] AS trend_daily_sales,
                col_combined["nl_avg_weekly_sales"] AS nl_avg_weekly_sales,
                col_combined["nl_single_weekly_sales"] AS nl_single_weekly_sales,
                col_combined["decay_25_nl_avg_weekly_sales"] AS decay_25_nl_avg_weekly_sales,
                col_combined["decay_50_nl_avg_weekly_sales"] AS decay_50_nl_avg_weekly_sales,
                col_combined["decay_75_nl_avg_weekly_sales"] AS decay_75_nl_avg_weekly_sales,
                col_combined["wt_avg_sum_sales"] AS wt_avg_sum_sales,
                col_combined["decay_25_wt_avg_sum_sales"] AS decay_25_wt_avg_sum_sales,
                col_combined["decay_50_wt_avg_sum_sales"] AS decay_50_wt_avg_sum_sales,
                col_combined["decay_75_wt_avg_sum_sales"] AS decay_75_wt_avg_sum_sales,
                col_combined["wt_single_sum_sales"] AS wt_single_sum_sales,
                col_combined["store_sales_wt"] AS store_sales_wt,
                col_combined["num_promo"] AS num_promo,
                col_combined["promo_date"] AS promo_date,
                col_combined["num_promo_all"] AS num_promo_all,
                col_combined["nl_avg_based_on"] AS nl_avg_based_on,
                col_combined["wt_avg_based_on"] AS wt_avg_based_on,
                col_combined["wt_avg_promo_type"] AS wt_avg_promo_type
                FROM (
                    SELECT
                    *,
                    EXPLODE(ARRAYS_ZIP(wt_avg_promo_type, daily_sales, decay_25_daily_sales, decay_50_daily_sales, decay_75_daily_sales, daily_sales_nl, decay_25_daily_sales_nl, decay_50_daily_sales_nl, decay_75_daily_sales_nl, trend_daily_sales, std_daily_sales, std_daily_sales_nl, decay_25_nl_avg_weekly_sales, decay_50_nl_avg_weekly_sales, decay_75_nl_avg_weekly_sales, nl_avg_weekly_sales, nl_single_weekly_sales, decay_25_wt_avg_sum_sales, decay_50_wt_avg_sum_sales, decay_75_wt_avg_sum_sales, wt_avg_sum_sales, wt_single_sum_sales, wt_avg_based_on, nl_avg_based_on, num_promo, num_promo_all, promo_date, store_sales_wt, avg_nl_price, decay_25_avg_nl_price, decay_50_avg_nl_price, decay_75_avg_nl_price, avg_wt_price, decay_25_avg_wt_price, decay_50_avg_wt_price, decay_75_avg_wt_price)) 
                    AS col_combined
                        FROM(
                            SELECT
                            *,           

                            --{'''DIFF_ARRAY(wt_price, avg_wt_price)   AS wt_diff_in_sales_price, DIFF_ARRAY(nl_price, avg_nl_price)  AS nl_diff_in_sales_price,
                            --DIFF_ARRAY(wt_price, decay_25_avg_wt_price)  AS wt_diff_25_in_sales_price, DIFF_ARRAY(nl_price, decay_25_avg_nl_price)  AS nl_diff_25_in_sales_price,
                            --DIFF_ARRAY(wt_price, decay_50_avg_wt_price)  AS wt_diff_50_in_sales_price, DIFF_ARRAY(nl_price, decay_50_avg_nl_price)  AS nl_diff_50_in_sales_price,
                            --DIFF_ARRAY(wt_price, decay_75_avg_wt_price)  AS wt_diff_75_in_sales_price, DIFF_ARRAY(nl_price, decay_75_avg_nl_price)  AS nl_diff_75_in_sales_price,
                            --''' if not self.predict_set else ''}


                                /* AVG SALE PROMO TYPE */
                            ARRAY(wt_avg_pred_sale_promo_type_decode, wt_avg_sim_sale_promo_type_decode, wt_avg_fambrand_based_on_decode, wt_avg_familygrp_based_on_decode)  AS wt_avg_promo_type
                            FROM(
                                    SELECT
                                        *,

                                        DECODE(wt_avg_pred_sale_promo_type, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS wt_avg_pred_sale_promo_type_decode,
                                        DECODE(wt_avg_sim_sale_promo_type, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS wt_avg_sim_sale_promo_type_decode,
                                        DECODE(wt_avg_fambrand_based_on, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS wt_avg_fambrand_based_on_decode,
                                        DECODE(wt_avg_familygrp_based_on, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS wt_avg_familygrp_based_on_decode,

                                            /* OS SALES */
                                        ARRAY(avg_preds_daily_sales, avg_sim_daily_sales, avg_fambrands_daily_sales, avg_familygrp_daily_sales) AS daily_sales,
                                        ARRAY(decay_25_avg_preds_daily_sales, decay_25_avg_sim_daily_sales, decay_25_avg_fambrands_daily_sales, decay_25_avg_familygrp_daily_sales) AS decay_25_daily_sales,
                                        ARRAY(decay_50_avg_preds_daily_sales, decay_50_avg_sim_daily_sales, decay_50_avg_fambrands_daily_sales, decay_50_avg_familygrp_daily_sales) AS decay_50_daily_sales,
                                        ARRAY(decay_75_avg_preds_daily_sales, decay_75_avg_sim_daily_sales, decay_75_avg_fambrands_daily_sales, decay_75_avg_familygrp_daily_sales) AS decay_75_daily_sales,
                                        ARRAY(avg_preds_daily_sales_nl, avg_sim_daily_sales_nl, avg_fambrands_daily_sales_nl, avg_familygrp_daily_sales_nl) AS daily_sales_nl,
                                        ARRAY(decay_25_avg_preds_daily_sales_nl, decay_25_avg_sim_daily_sales_nl, decay_25_avg_fambrands_daily_sales_nl, decay_25_avg_familygrp_daily_sales_nl) AS decay_25_daily_sales_nl,
                                        ARRAY(decay_50_avg_preds_daily_sales_nl, decay_50_avg_sim_daily_sales_nl, decay_50_avg_fambrands_daily_sales_nl, decay_50_avg_familygrp_daily_sales_nl) AS decay_50_daily_sales_nl,
                                        ARRAY(decay_75_avg_preds_daily_sales_nl, decay_75_avg_sim_daily_sales_nl, decay_75_avg_fambrands_daily_sales_nl, decay_75_avg_familygrp_daily_sales_nl) AS decay_75_daily_sales_nl,

                                            /* TREND SALES */
                                        ARRAY(trend_preds_daily_sales, trend_fambrands_daily_sales, trend_familygrp_daily_sales) AS trend_daily_sales,

                                            /* STANDARD DEVIATIONS */
                                        ARRAY(std_preds_daily_sales, std_sim_daily_sales, std_fambrands_daily_sales, std_familygrp_daily_sales) AS std_daily_sales,
                                        ARRAY(std_preds_daily_sales_nl, std_sim_daily_sales_nl, std_fambrands_daily_sales_nl, std_familygrp_daily_sales_nl) AS std_daily_sales_nl,


                                            /* NL SINGLE & AVG WEEKLY SALES */
                                        ARRAY(decay_25_nl_avg_preds_weekly_sales, decay_25_nl_avg_sim_weekly_sales,
                                            decay_25_nl_avg_fambrand_weekly_sales, decay_25_nl_avg_familygrp_weekly_sales) AS decay_25_nl_avg_weekly_sales,
                                        ARRAY(decay_50_nl_avg_preds_weekly_sales, decay_50_nl_avg_sim_weekly_sales,
                                            decay_50_nl_avg_fambrand_weekly_sales, decay_50_nl_avg_familygrp_weekly_sales) AS decay_50_nl_avg_weekly_sales,
                                        ARRAY(decay_75_nl_avg_preds_weekly_sales, decay_75_nl_avg_sim_weekly_sales,
                                            decay_75_nl_avg_fambrand_weekly_sales, decay_75_nl_avg_familygrp_weekly_sales) AS decay_75_nl_avg_weekly_sales,
                                        ARRAY(nl_avg_preds_weekly_sales, nl_avg_sim_weekly_sales, nl_avg_fambrand_weekly_sales, nl_avg_familygrp_weekly_sales) AS nl_avg_weekly_sales,
                                        ARRAY(nl_single_pred_avg_weekly_sales, nl_single_sim_avg_weekly_sales, nl_avg_fambrand_weekly_sales, nl_avg_familygrp_weekly_sales) AS nl_single_weekly_sales,

                                            /* WT SINGLE & AVG SUM SALES */
                                        ARRAY(wt_decay_25_avg_pred_sum_sales_first_wt, decay_25_avg_sim_sum_sales_first_wt,
                                            decay_25_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_25_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_25_wt_avg_sum_sales,
                                        ARRAY(wt_decay_50_avg_pred_sum_sales_first_wt, decay_50_avg_sim_sum_sales_first_wt,
                                            decay_50_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_50_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_50_wt_avg_sum_sales,
                                        ARRAY(wt_decay_75_avg_pred_sum_sales_first_wt, decay_75_avg_sim_sum_sales_first_wt,
                                            decay_75_wt_avg_fambrand_sum_sales_first_wt_hhz, decay_75_wt_avg_familygrp_sum_sales_first_wt_hhz) AS decay_75_wt_avg_sum_sales,
                                        ARRAY(wt_avg_pred_sum_sales_first_wt, wt_avg_sim_sum_sales_first_wt, wt_avg_fambrand_sum_sales_first_wt_hhz, wt_avg_familygrp_sum_sales_first_wt_hhz) AS wt_avg_sum_sales,
                                        ARRAY(wt_single_pred_sum_sales_first_wt, wt_single_sim_sum_sales_first_wt, wt_avg_fambrand_sum_sales_first_wt_hhz, wt_avg_familygrp_sum_sales_first_wt_hhz) AS wt_single_sum_sales,

                                            /* based on */
                                        ARRAY(1,2,3,4) AS wt_avg_based_on,
                                        ARRAY(1,2,3,4) AS nl_avg_based_on,

                                            /* NUMBER OF PROMOTIONS */
                                        ARRAY(wt_pred_num_promo, wt_sim_num_promo)  AS num_promo,
                                        ARRAY(wt_pred_num_promo_all, wt_sim_num_promo_all)  AS num_promo_all,

                                            /* PROMOTION DATE */
                                        ARRAY(wt_pred_promo_date, wt_sim_promo_date) AS promo_date,

                                            /* STORE SALES */ 
                                        ARRAY(avg_preds_d7_qty_sales_sum, avg_fambrand_d7_qty_sales_sum) AS store_sales_wt,

                                            /* PRICE */
                                        ARRAY(nl_avg_preds_sales_price, nl_avg_sim_sales_price, nl_avg_fambrand_sales_price, nl_avg_familygrp_sales_price) AS avg_nl_price,
                                        ARRAY(decay_25_nl_avg_preds_sales_price, decay_25_nl_avg_sim_sales_price, decay_25_nl_avg_fambrand_sales_price, decay_25_nl_avg_familygrp_sales_price) AS decay_25_avg_nl_price,
                                        ARRAY(decay_50_nl_avg_preds_sales_price, decay_50_nl_avg_sim_sales_price, decay_50_nl_avg_fambrand_sales_price, decay_50_nl_avg_familygrp_sales_price) AS decay_50_avg_nl_price,
                                        ARRAY(decay_75_nl_avg_preds_sales_price, decay_75_nl_avg_sim_sales_price, decay_75_nl_avg_fambrand_sales_price, decay_75_nl_avg_familygrp_sales_price) AS decay_75_avg_nl_price,
                                        ARRAY(wt_avg_pred_wt_price, wt_avg_sim_wt_price) AS avg_wt_price,
                                        ARRAY(decay_25_wt_avg_pred_wt_price, decay_25_wt_avg_sim_wt_price) AS decay_25_avg_wt_price,
                                        ARRAY(decay_50_wt_avg_pred_wt_price, decay_50_wt_avg_sim_wt_price) AS decay_50_avg_wt_price,
                                        ARRAY(decay_75_wt_avg_pred_wt_price, decay_75_wt_avg_sim_wt_price) AS decay_75_avg_wt_price

                                    FROM inference_data
                                    )
                                )
                            )   
                         """
                     )
        
    def _preprocessing_pandas(self):
        #ToDo: no promotion_medium_type, sell_off_horizon etc. if self.predict_set = True
        
        inference_df = self.feature_df.load_df().toPandas()
                
        # Define a function to encode the promo type valuesx
        inference_df= (inference_df
        # Apply the necessary transformations to each row in the DataFrame
        .assign(daily_sales=lambda x: x['avg_preds_daily_sales'].fillna(x['avg_fambrands_daily_sales']).fillna(x['avg_familygrp_daily_sales']),
                daily_sales_nl=lambda x: x['avg_preds_daily_sales_nl'].fillna(x['avg_fambrands_daily_sales_nl']).fillna(x['avg_familygrp_daily_sales_nl']),
                trend_daily_sales=lambda x:  x['trend_fambrands_daily_sales'].fillna(x['trend_familygrp_daily_sales']),
                nl_avg_weekly_sales=lambda x: x['nl_single_pred_avg_weekly_sales'].fillna(x['nl_single_sim_avg_weekly_sales']).fillna(x['nl_avg_fambrand_weekly_sales']).fillna(x['nl_avg_familygrp_weekly_sales']),
                decay_nl_avg_weekly_sales=lambda x: x['decay_nl_avg_preds_weekly_sales'].fillna(x['decay_nl_avg_sim_weekly_sales']).fillna(x['decay_nl_avg_fambrand_weekly_sales']).fillna(x['decay_nl_avg_familygrp_weekly_sales']),
                nl_single_weekly_sales=lambda x:  x['nl_single_pred_avg_weekly_sales'].fillna(x['nl_single_sim_avg_weekly_sales']).fillna(x['nl_avg_fambrand_weekly_sales']).fillna(x['nl_avg_familygrp_weekly_sales']),
                wt_avg_sum_sales=lambda x:  x['wt_avg_pred_sum_sales_first_wt'].fillna(x['wt_avg_sim_sum_sales_first_wt']).fillna(x['wt_avg_fambrand_sum_sales_first_wt_hhz']).fillna(x['wt_avg_familygrp_sum_sales_first_wt_hhz']),
                decay_wt_avg_sum_sales=lambda x:  x['wt_decay_avg_pred_sum_sales_first_wt'].fillna(x['wt_decay_avg_sim_sum_sales_first_wt']).fillna(x['decay_wt_avg_fambrand_sum_sales_first_wt_hhz']).fillna(x['decay_wt_avg_familygrp_sum_sales_first_wt_hhz']),
                wt_single_sum_sales=lambda x:  x['wt_single_pred_sum_sales_first_wt'].fillna(x['wt_single_sim_sum_sales_first_wt']).fillna(x['wt_avg_fambrand_sum_sales_first_wt_hhz']).fillna(x['wt_avg_familygrp_sum_sales_first_wt_hhz']),
                wt_single_promo_type=lambda x:  x[['wt_single_pred_sale_promo_type', 'wt_single_sim_sale_promo_type', 'wt_avg_fambrand_based_on', 'wt_avg_familygrp_based_on']].apply(lambda x: np.nan if x.isnull().all() else x.dropna().iloc[0], axis=1).replace(self.mapping),
                wt_avg_promo_type=lambda x:  x[['wt_avg_pred_sale_promo_type', 'wt_avg_sim_sale_promo_type', 'wt_avg_fambrand_based_on', 'wt_avg_familygrp_based_on']].apply(lambda x: np.nan if x.isnull().all() else x.dropna().iloc[0], axis=1).replace(self.mapping),
                store_sales_wt=lambda x:  x['avg_preds_d7_qty_sales_sum'].fillna(x['avg_fambrand_d7_qty_sales_sum']),
                sales_price_nl=lambda x:  x['nl_avg_preds_sales_price'].fillna(x['nl_avg_fambrand_sales_price']).fillna(x['nl_avg_familygrp_sales_price']),
                # WT & NL SINGLE ITEM_IAN
                wt_single_item_ian=lambda x: x['wt_single_pred_item_ian'].fillna(x['wt_single_sim_item_ian']),
                nl_single_item_ian=lambda x: x['nl_single_pred_item_ian'].fillna(x['nl_single_sim_item_ian']),
                diff_in_sales_price=lambda x: x['sales_price_nl'] - x['sales_price']
                                                           )
                      )
        source_df = inference_df.copy()
        # NL SINGLE & AVG BASED ON, ITEMS USED
        source_df['nl_avg_based_on'] = np.select([source_df['nl_avg_preds_weekly_sales'].notnull(),
                                                  source_df['nl_avg_sim_weekly_sales'].notnull(),
                                                  source_df['nl_avg_fambrand_weekly_sales'].notnull(),
                                                  source_df['nl_avg_familygrp_weekly_sales'].notnull()],
                                                 [1, 2, 3, 4], default=0)
        source_df['nl_single_based_on'] = np.select([source_df['nl_single_pred_avg_weekly_sales'].notnull(),
                                                     source_df['nl_single_sim_avg_weekly_sales'].notnull(),
                                                     source_df['nl_avg_fambrand_weekly_sales'].notnull(),
                                                     source_df['nl_avg_familygrp_weekly_sales'].notnull()],
                                                    [1, 2, 3, 4], default=0)
        source_df['nl_items_used'] = np.select([source_df['nl_avg_preds_weekly_sales'].notnull(),
                                                 source_df['nl_avg_sim_weekly_sales'].notnull()],
                                                [source_df['nl_pred_items_used_in_avg'],
                                                 source_df['nl_sim_items_used_in_avg']], default=None)

        # WT SINGLE & AVG BASED ON, ITEMS USED
        source_df['wt_avg_based_on'] = np.select([source_df['wt_avg_pred_sum_sales_first_wt'].notnull(),
                                                  source_df['wt_avg_sim_sum_sales_first_wt'].notnull(),
                                                  source_df['wt_avg_fambrand_sum_sales_first_wt_hhz'].notnull(),
                                                  source_df['wt_avg_familygrp_sum_sales_first_wt_hhz'].notnull()],
                                                 [1, 2, 3, 4], default=0)
        source_df['wt_single_based_on'] = np.select([source_df['wt_single_pred_sum_sales_first_wt'].notnull(),
                                                     source_df['wt_avg_sim_sum_sales_first_wt'].notnull(),
                                                     source_df['wt_avg_fambrand_sum_sales_first_wt_hhz'].notnull(),
                                                     source_df['wt_avg_familygrp_sum_sales_first_wt_hhz'].notnull()],
                                                    [1, 2, 3, 4], default=0)
        source_df['wt_items_used'] = np.select([source_df['wt_avg_pred_sum_sales_first_wt'].notnull(),
                                                 source_df['wt_avg_sim_sum_sales_first_wt'].notnull()],
                                                [source_df['wt_pred_items_used_in_avg'],
                                                 source_df['wt_sim_items_used_in_avg']], default=None)
        
        return source_df

# COMMAND ----------

class ModelWorkflow:
    def __init__(
        self,
        wshop_cd: List[str],
        model_name: str,
        df: pd.DataFrame,
        input_columns: List[str],
        test_range_plan_cd: List[int],
        loss_metric: Any = 'RMSE',
        confidence: float = 0.90,
        log_transform_target: bool = False,
        noise_detector: bool = False,
        batch_size: int = 250,
        subsample: float = None,
        overfit_error: float = 11,
        params_det: dict = None
    ):
        '''
        In the constructor main parameters are initialized 

        Args:
            wshop_cd (str): country of the shop
            model_name (str): name of the model (e.g. paf_promo_de)
            df (pd.DataFrame): training dataset
            input_columns (List[str]): list of the features used in the training 
            test_range_plan_cd (List[str]): Time ranges for test data
            loss_metric (Any): loss-function used in the training by default 'RMSE'
            confidence (float): confidence level for the uncertainty model by default 0.9
            log_transform_target (bool): taking log of the target sales with further application of the exp function by default False
        Returns:
            calling function to split data in train/test
            calling function to calculate baseline and manager metrics
        '''
        self.model_name = model_name
        self.wshop_cd = wshop_cd
        self.input_columns = input_columns
        self.test_range_plan_cd = test_range_plan_cd
        self.confidence = confidence
        self.loss_metric = loss_metric
        
        self.dict_train_test = None
        self.dict_fitted_all = None
        self.fitted_on_all_data = False
        self.tuned = False
        self.log_y = log_transform_target
        if (batch_size is not None) & (subsample is not None):
            raise ValueError("Use either batch_size or subsample!")
        if (batch_size is None) & (subsample is None):
            raise ValueError("Specify either batch_size or subsample!")
        self.batch_size = batch_size
        self.subsample = subsample
        self.overfit_error = overfit_error
        self.noise_detector = noise_detector
    
        # define model type
        self.promo_model = True if 'after' not in model_name else False


        if self.promo_model:
            self.manager_qty = 'promo_qty'
            self.baseline_qty = 'baseline_wt'
            self.target = "target_wt"
            dropna_cols = ['wt_avg_based_on', 'wt_avg_promo_type']
            self.cat_features = ["promotion_medium_type", "wt_avg_based_on", "wt_avg_promo_type", "tv_fg", "brand_type_cd"]
            if "wshop_cd" in self.input_columns:
                self.cat_features.append("wshop_cd")
            self.based_on = 'wt_avg_based_on'
            self.sales_column = 'wt_avg_sum_sales'
            self.sales_feature = [col for col in self.input_columns if 'wt_avg_sum_sales' in col]
        else:
            self.manager_qty = 'after_promo_qty'
            self.baseline_qty = 'baseline_nl'
            self.target = "target_nl"
            dropna_cols = ['nl_avg_based_on']
            self.cat_features = ["promotion_medium_type", "nl_avg_based_on", "tv_fg", "brand_type_cd"]
            if "wshop_cd" in self.input_columns:
                self.cat_features.append("wshop_cd")
            self.based_on = 'nl_avg_based_on'
            self.sales_column = 'nl_avg_weekly_sales'
            self.sales_feature = [col for col in self.input_columns if 'nl_avg_weekly_sales' in col]

        # throw out datapoints which should not be used for training and merge baseline values
        self.df = (
        df.dropna(subset=dropna_cols+[self.target])
            .query(f'range_plan_cd <= {self.test_range_plan_cd[-1]}')
            .merge(get_baseline_nl(tuple(self.wshop_cd))[['wshop_cd', 'item_ian', 'range_plan_cd', 'baseline_nl']], on=['wshop_cd', 'item_ian', 'range_plan_cd'], how='left')
            .merge(get_baseline_wt(tuple(self.wshop_cd))[['wshop_cd', 'item_ian', 'range_plan_cd', 'baseline_wt']], on=['wshop_cd', 'item_ian', 'range_plan_cd'], how='left')
            .assign(partition=lambda x: ['TRAIN' if i<self.test_range_plan_cd[0] else 'TEST' for i in x.range_plan_cd],
                    after_promo_qty=lambda x: x.after_promo_qty / (x.sell_off_horizon-3))
            )
        
        # split data into train, test
        self._train_test_val_split(test_range_plan_cd=self.test_range_plan_cd)
        # define if model is trained on data of all countries
        self.all_countries = True if len(self.train["wshop_cd"].unique()) != 1 else False
        # indices of categorical features
        self.cols_index = [self.X_train.columns.get_loc(col) for col in self.cat_features]
        # setting base params for the model
        self.base_params = {
            'allow_writing_files': False,
            'random_state': 123,
            'cat_features': self.cols_index,
            'loss_function': self.loss_metric
        }
        # indices of noisy training data
        if self.noise_detector:
            self.noisy_index = self.objective_importance(params = params_det if params_det else None, size_sample = self.subsample, batch_size = self.batch_size, overfit_error = self.overfit_error)
        else:
            self.noisy_index = None
        # calculate baseline and manager metrics
        self._calculate_baselines()
        

        
    def _train_test_val_split(self, test_range_plan_cd: List = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Splitting input df into train and test data based on provided time periods. Defining datasets for subgroups (predecessor, no_predecessor, most_sold, least_sold)

        Args:
            test_range_plan_cd (list): Time ranges for test data

        Returns:
            splitted X_test y_test and subgroups datasets
        '''
        self.train = self.df.query('partition == "TRAIN"').sort_values(by=['range_plan_cd']).reset_index(drop=True)
        self.train_pred = self.train.query(f'{self.based_on} == 1')
        self.train_no_pred = self.train.query(f'{self.based_on} != 1')
        self.train_most_sold = self.train.query(f'{self.sales_column} > {self.sales_column}.quantile(0.85)')
        self.train_least_sold = self.train.query(f'{self.sales_column} < {self.sales_column}.quantile(0.15)')

        self.train_wshop = self.df.query(f'(partition == "TRAIN") and (wshop_cd.isin({self.wshop_cd}))').sort_values(by=['range_plan_cd']).reset_index(drop=True)
        self.train_pred_wshop = self.train_wshop.query(f'{self.based_on} == 1')
        self.train_no_pred_wshop = self.train_wshop.query(f'{self.based_on} != 1')
        self.train_most_sold_wshop = self.train_wshop.query(f'{self.sales_column} > {self.sales_column}.quantile(0.85)')
        self.train_least_sold_wshop = self.train_wshop.query(f'{self.sales_column} < {self.sales_column}.quantile(0.15)')


        self.test = self.df.query(f'(partition == "TEST") and (wshop_cd.isin({self.wshop_cd}))').sort_values(by=['range_plan_cd']).reset_index(drop=True)
        self.test_all = self.df.query(f'partition == "TEST"').sort_values(by=['range_plan_cd']).reset_index(drop=True)
        self.test_pred = self.test.query(f'{self.based_on} == 1')
        self.test_no_pred = self.test.query(f'{self.based_on} != 1')
        self.test_most_sold = self.test.query(f'{self.sales_column} > {self.sales_column}.quantile(0.85)')
        self.test_least_sold = self.test.query(f'{self.sales_column} < {self.sales_column}.quantile(0.15)')

        self.X_train, self.y_train = self.train[[x for x in self.input_columns]], self.train[self.target]
        self.X_train_wshop, self.y_train_wshop = self.train_wshop[[x for x in self.input_columns]], self.train_wshop[self.target]
        self.X_test, self.y_test = self.test[[x for x in self.input_columns]], self.test[self.target]
        self.X_test_all, self.y_test_all = self.test_all[[x for x in self.input_columns]], self.test_all[self.target]
            
        self.X = self.df[[x for x in self.input_columns]]
        self.y = self.df[self.target]
        self.y_wshop = self.df.query(f'wshop_cd.isin({self.wshop_cd})')[self.target]

    def _calculate_baselines(self):
        '''
        Calculating metrics from the baseline's and manager's prospective. Also calculation of the expected values

        Args:
            None

        Returns:
            Regression metrics such as(mae, rmse, mape)
        '''
        # removing noisy training data
        if self.noisy_index:
            self.train = self.train[~self.train.index.isin(self.noisy_index)]
            self.train_pred = self.train_pred[~self.train_pred.index.isin(self.noisy_index)]
            self.train_no_pred = self.train_no_pred[~self.train_no_pred.index.isin(self.noisy_index)]
            self.train_most_sold = self.train_most_sold[~self.train_most_sold.index.isin(self.noisy_index)]
            self.train_least_sold = self.train_least_sold[~self.train_least_sold.index.isin(self.noisy_index)]
            self.filtered = True
        
        test_baseline = []
        test_pred_baseline = []
        test_no_pred_baseline = []
        test_most_sold_baseline = []
        test_least_sold_baseline = []
        test_manager = []
        test_pred_manager = []
        test_no_pred_manager = []
        test_most_sold_manager = []
        test_least_sold_manager = []
        test_expected_val = []
        test_pred_expected_val = []
        test_no_pred_expected_val = []
        test_most_sold_expected_val = []
        test_least_sold_expected_val = []
        self.metrics = {}
        for num, i in enumerate(self.wshop_cd):

            test_baseline.append(self.test.query(f'wshop_cd == "{i}"')[[self.target, self.baseline_qty]].dropna())
            test_pred_baseline.append(self.test_pred.query(f'wshop_cd == "{i}"')[[self.target, self.baseline_qty]].dropna())
            test_no_pred_baseline.append(self.test_no_pred.query(f'wshop_cd == "{i}"')[[self.target, self.baseline_qty]].dropna())
            test_most_sold_baseline.append(self.test_most_sold.query(f'wshop_cd == "{i}"')[[self.target, self.baseline_qty]].dropna())
            test_least_sold_baseline.append(self.test_least_sold.query(f'wshop_cd == "{i}"')[[self.target, self.baseline_qty]].dropna())

            test_manager.append(self.test.query(f'wshop_cd == "{i}"')[[self.target, self.manager_qty]].dropna())
            test_pred_manager.append(self.test_pred.query(f'wshop_cd == "{i}"')[[self.target, self.manager_qty]].dropna())
            test_no_pred_manager.append(self.test_no_pred.query(f'wshop_cd == "{i}"')[[self.target, self.manager_qty]].dropna())
            test_most_sold_manager.append(self.test_most_sold.query(f'wshop_cd == "{i}"')[[self.target, self.manager_qty]].dropna())
            test_least_sold_manager.append(self.test_least_sold.query(f'wshop_cd == "{i}"')[[self.target, self.manager_qty]].dropna())

            test_expected_val.append([np.mean(self.train.query(f'wshop_cd == "{i}"')[self.target])]*len(self.test.query(f'wshop_cd == "{i}"')))
            test_pred_expected_val.append([np.mean(self.train_pred.query(f'wshop_cd == "{i}"')[self.target])]*len(self.test_pred.query(f'wshop_cd == "{i}"')))
            test_no_pred_expected_val.append([np.mean(self.train_no_pred.query(f'wshop_cd == "{i}"')[self.target])]*len(self.test_no_pred.query(f'wshop_cd == "{i}"')))
            test_most_sold_expected_val.append([np.mean(self.train_most_sold.query(f'wshop_cd == "{i}"')[self.target])]*len(self.test_most_sold.query(f'wshop_cd == "{i}"')))
            test_least_sold_expected_val.append([np.mean(self.train_least_sold.query(f'wshop_cd == "{i}"')[self.target])]*len(self.test_least_sold.query(f'wshop_cd == "{i}"')))

        
            self.metrics[f'{i}'] = {
                'expected_value': {
                    # test metrics
                    'test_mae':                   mean_absolute_error(self.test.query(f'wshop_cd == "{i}"')[self.target], test_expected_val[num]),
                    'test_mae_predecessor':       mean_absolute_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], test_pred_expected_val[num]),
                    'test_mae_no_predecessor':    mean_absolute_error(self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target], test_no_pred_expected_val[num]),
                    #'test_mae_most_sold':         mean_absolute_error(self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target], test_most_sold_expected_val[num]),
                    #'test_mae_least_sold':        mean_absolute_error(self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target], test_least_sold_expected_val[num]),
                    'test_mae_perc':              mean_absolute_error(self.test.query(f'wshop_cd == "{i}"')[self.target], test_expected_val[num]) / np.mean(self.test.query(f'wshop_cd == "{i}"')[self.target]),
                    'test_mae_predecessor':       mean_absolute_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], test_pred_expected_val[num]),
                    'test_mae_perc_predecessor':  mean_absolute_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], test_pred_expected_val[num]) / np.mean(self.test_pred.query(f'wshop_cd == "{i}"')[self.target]),
                    'test_mape':                  mean_absolute_percentage_error(self.test.query(f'wshop_cd == "{i}"')[self.target], test_expected_val[num]),
                    'test_rmse':                  mean_squared_error(self.test.query(f'wshop_cd == "{i}"')[self.target], test_expected_val[num], squared=False),
                    'test_rmse_predecessor':      mean_squared_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], test_pred_expected_val[num], squared=False),
                    'test_rmse_no_predecessor':   mean_squared_error(self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target], test_no_pred_expected_val[num], squared=False),
                    #'test_rmse_most_sold':        mean_squared_error(self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target], test_most_sold_expected_val[num], squared=False),
                    #'test_rmse_least_sold':       mean_squared_error(self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target], test_least_sold_expected_val[num], squared=False),
                    'test_mape_predecessor':      mean_absolute_percentage_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], test_pred_expected_val[num]),

                },
                # baseline metrics
                'baseline': {
                    'test_mae':                   mean_absolute_error(test_baseline[num][self.target], test_baseline[num][self.baseline_qty]),
                    'test_mae_predecessor':       mean_absolute_error(test_pred_baseline[num][self.target], test_pred_baseline[num][self.baseline_qty]),
                    'test_mae_no_predecessor':    mean_absolute_error(test_no_pred_baseline[num][self.target], test_no_pred_baseline[num][self.baseline_qty]),
                    'test_mae_most_sold':         mean_absolute_error(test_most_sold_baseline[num][self.target], test_most_sold_baseline[num][self.baseline_qty]),
                    'test_mae_least_sold':        mean_absolute_error(test_least_sold_baseline[num][self.target], test_least_sold_baseline[num][self.baseline_qty]),
                    'test_rmse':                  mean_squared_error(test_baseline[num][self.target], test_baseline[num][self.baseline_qty], squared=False),
                    'test_rmse_predecessor':      mean_squared_error(test_pred_baseline[num][self.target], test_pred_baseline[num][self.baseline_qty], squared=False),
                    'test_rmse_no_predecessor':   mean_squared_error(test_no_pred_baseline[num][self.target], test_no_pred_baseline[num][self.baseline_qty], squared=False),
                    'test_rmse_most_sold':        mean_squared_error(test_most_sold_baseline[num][self.target], test_most_sold_baseline[num][self.baseline_qty], squared=False),
                    'test_rmse_least_sold':       mean_squared_error(test_least_sold_baseline[num][self.target], test_least_sold_baseline[num][self.baseline_qty], squared=False),
                    'test_mae_perc':              mean_absolute_error(test_baseline[num][self.target], test_baseline[num][self.baseline_qty]) / np.mean(test_baseline[num][self.target]),
                    'test_mae_predecessor':       mean_absolute_error(test_pred_baseline[num][self.target], test_pred_baseline[num][self.baseline_qty]),
                    'test_mae_perc_predecessor':  mean_absolute_error(test_pred_baseline[num][self.target], test_pred_baseline[num][self.baseline_qty])/ np.mean(test_pred_baseline[num][self.target]),
                    'test_mape':                  mean_absolute_percentage_error(test_baseline[num][self.target], test_baseline[num][self.baseline_qty]),
                    'test_mape_predecessor':      mean_absolute_percentage_error(test_pred_baseline[num][self.target], test_pred_baseline[num][self.baseline_qty])
                    
                },
                # # manager metrics
                'manager': {
                    'test_mae':                   mean_absolute_error(test_manager[num][self.target], test_manager[num][self.manager_qty]),
                    'test_mae_perc':              mean_absolute_error(test_manager[num][self.target], test_manager[num][self.manager_qty]) / np.mean(test_manager[num][self.target]),
                    'test_mae_predecessor':       mean_absolute_error(test_pred_manager[num][self.target], test_pred_manager[num][self.manager_qty]),
                    'test_mae_no_predecessor':    mean_absolute_error(test_no_pred_manager[num][self.target], test_no_pred_manager[num][self.manager_qty]),
                    'test_mae_most_sold':         mean_absolute_error(test_most_sold_manager[num][self.target], test_most_sold_manager[num][self.manager_qty]),
                    'test_mae_least_sold':        mean_absolute_error(test_least_sold_manager[num][self.target], test_least_sold_manager[num][self.manager_qty]),
                    'test_mae_perc_predecessor':  mean_absolute_error(test_pred_manager[num][self.target], test_pred_manager[num][self.manager_qty])/ np.mean(test_pred_manager[num][self.target]),
                    'test_mape':                  mean_absolute_percentage_error(test_manager[num][self.target], test_manager[num][self.manager_qty]),
                    'test_mape_predecessor':      mean_absolute_percentage_error(test_pred_manager[num][self.target], test_pred_manager[num][self.manager_qty]),
                    'test_rmse':                  mean_squared_error(test_manager[num][self.target], test_manager[num][self.manager_qty], squared=False),
                    'test_rmse_predecessor':      mean_squared_error(test_pred_manager[num][self.target], test_pred_manager[num][self.manager_qty], squared=False),
                    'test_rmse_no_predecessor':   mean_squared_error(test_no_pred_manager[num][self.target], test_no_pred_manager[num][self.manager_qty], squared=False),
                    'test_rmse_most_sold':        mean_squared_error(test_most_sold_manager[num][self.target], test_most_sold_manager[num][self.manager_qty], squared=False),
                    'test_rmse_least_sold':       mean_squared_error(test_least_sold_manager[num][self.target], test_least_sold_manager[num][self.manager_qty], squared=False)
                },
                }
    
        
    def _update_metrics_new(self, predictions):
            '''
            Calculating metrics for test and train data based on the built model

            Args:
                preds: dictionary of predictions made for test and train data

            Returns:
                preds_dict (dict): dictionary of calculated matrics 
            '''
            preds_dict = {}
            for i in self.wshop_cd:
                preds = predictions[f'{i}']
                # if fitted on all data
                y_train = self.df.query(f'wshop_cd == "{i}"')[self.target] if self.fitted_on_all_data else self.train.query(f'wshop_cd == "{i}"')[self.target]
        
                preds_dict[f'{i}'] =  {
                    'model':{
                        "train_mae": mean_absolute_error(preds["train"], self.train.query(f'wshop_cd == "{i}"')[self.target]),
                        "train_rmse": mean_squared_error(preds["train"], self.train.query(f'wshop_cd == "{i}"')[self.target], squared=False)
                    }
                }
                # if there are test data
                if not self.fitted_on_all_data:
                    preds_dict[f'{i}'] = {
                    'model': { 
                        "test_mae": mean_absolute_error(preds["test"], self.test.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_mae_perc":  mean_absolute_error(preds["test"], self.test.query(f'wshop_cd == "{i}"')[self.target]) / np.mean(self.test.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_mae_predecessor": mean_absolute_error(preds["test_predecessor"], self.test_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_mae_perc_predecessor": mean_absolute_error(preds["test_predecessor"], self.test_pred.query(f'wshop_cd == "{i}"')[self.target]) / np.mean(self.test_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        
                        "test_mape": mean_absolute_percentage_error(self.test.query(f'wshop_cd == "{i}"')[self.target], preds["test"]),
                        #"test_mape_perc":  mean_absolute_percentage_error(self.test[self.target], preds["test"]) / np.mean(self.test[self.target]) ,
                        "test_mape_predecessor": mean_absolute_percentage_error(self.test_pred.query(f'wshop_cd == "{i}"')[self.target], preds["test_predecessor"]),
                        #"test_mape_perc_predecessor":  mean_absolute_percentage_error(self.test_pred[self.target], preds["test_predecessor"]) / np.mean(self.test_pred[self.target])
                        # "test_bias": (preds["test"] - self.y_test).mean(),
                        "test_rmse": mean_squared_error(preds["test"], self.test.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        # "test_rmse_perc": mean_squared_error(preds["test"], self.y_test, squared=False) / np.mean(self.y_test),
                        "test_mean(mae,rmse)": np.mean([mean_absolute_error(preds["test"], self.test.query(f'wshop_cd == "{i}"')[self.target]), mean_squared_error(preds["test"], self.test.query(f'wshop_cd == "{i}"')[self.target], squared=False)]),
                        # 
                        #"test_mae_most_sold": mean_absolute_error(preds["test_most_sold"], self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"test_mae_perc_most_sold": mean_absolute_error(preds["test_most_sold"], self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target]) / np.mean(self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"test_rmse_most_sold": mean_squared_error(preds["test_most_sold"], self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        #"test_rmse_perc_most_sold": mean_squared_error(preds["test_most_sold"], self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False) / np.mean(self.test_most_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"test_mae_least_sold": mean_absolute_error(preds["test_least_sold"], self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"test_mae_perc_least_sold" : mean_absolute_error(preds["test_least_sold"], self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target]) / np.mean(self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"test_rmse_least_sold": mean_squared_error(preds["test_least_sold"], self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        #"test_rmse_perc_least_sold": mean_squared_error(preds["test_least_sold"], self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False) / np.mean(self.test_least_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_mae_no_predecessor":  mean_absolute_error(preds["test_no_predecessor"], self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_mae_perc_no_predecessor": mean_absolute_error(preds["test_no_predecessor"], self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target]) / np.mean(self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_rmse_no_predecessor":  mean_squared_error(preds["test_no_predecessor"], self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        "test_rmse_perc_no_predecessor": mean_squared_error(preds["test_no_predecessor"], self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target], squared=False) / np.mean(self.test_no_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "test_rmse_predecessor": mean_squared_error(preds["test_predecessor"], self.test_pred.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        # Train Metrics
                        "train_mae": mean_absolute_error(preds["train"], self.train.query(f'wshop_cd == "{i}"')[self.target]),
                        "train_rmse": mean_squared_error(preds["train"], self.train.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        #"train_mae_most_sold": mean_absolute_error(preds["train_most_sold"], self.train_most_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"train_rmse_most_sold": mean_squared_error(preds["train_most_sold"], self.train_most_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        #"train_mae_least_sold": mean_absolute_error(preds["train_least_sold"], self.train_least_sold.query(f'wshop_cd == "{i}"')[self.target]),
                        #"train_rmse_least_sold": mean_squared_error(preds["train_least_sold"], self.train_least_sold.query(f'wshop_cd == "{i}"')[self.target], squared=False),
                        "train_mae_predecessor": mean_absolute_error(preds["train_predecessor"], self.train_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "train_rmse_predecessor": mean_squared_error(preds["train_predecessor"], self.train_pred.query(f'wshop_cd == "{i}"')[self.target], squared=False),                
                        "train_mae_no_predecessor": mean_absolute_error(preds["train_no_predecessor"], self.train_no_pred.query(f'wshop_cd == "{i}"')[self.target]),
                        "train_rmse_no_predecessor": mean_squared_error(preds["train_no_predecessor"], self.train_no_pred.query(f'wshop_cd == "{i}"')[self.target], squared=False),                          
                    }
                    }
                   
            return preds_dict

            
    def _scor_func(self, y_actual, y_pred):
        '''
        Function is used to calculate average between mae and rmse in order to do further parameter optimization

        Args:
            y_actual: actual values
            y_pred: predicted values
        
        Returns:
            score: average between mae and rmse
        '''
        if self.log_y:
            y_actual = np.exp(y_actual)
            y_pred = np.exp(y_pred)
        else:
            pass
        rmse = -mean_squared_error(y_actual, y_pred, squared=False)
        mae = -mean_absolute_error(y_actual, y_pred)
        score = np.mean([rmse, mae])
        return score


    def _base_model_objective(self, delta_param : float):
        '''
        Objective function for delta parameter optimization of a Huber loss-function

        Args:
            delta_param (float): value of the delta parameter

        Returns:
            loss: calculated average between rmse and mae 
            status: status of the optimization iteration 
        '''
        # Creating base parameters for Huber loss-function
        huber_base_params = self.base_params.copy()
        huber_base_params.update({"loss_function":f'Huber:delta={str(delta_param)}'})
        # Creating regressor
        reg_obj = ctb.CatBoostRegressor(
           ** huber_base_params
        )
        # Wrapp custom score function
        model_score = make_scorer(self._scor_func, greater_is_better=False)
        # Creating timeseries split
        tcsv = TimeSeriesSplit(n_splits = self.cv, gap = 0)
        # Cross Validation
        loss_dict = cross_validate(
            reg_obj,
            self.X_train,
            np.log(self.y_train) if self.log_y else self.y_train,
            scoring=model_score,
            cv=tcsv,
            return_train_score=False,
        )
        # Loss value
        loss = np.mean(loss_dict["test_score"])

        return {"loss": loss, "status": STATUS_OK}    

    
    def find_best_huber_param(
        self,
        delta_space: float,
        max_evals: int = 64,
        #parallelism_value: int = 8,
        log_model: bool = False,
        register_model: bool = False,
        cv: int = 3
    ):
        '''
        Function for optimizing delta parameter in untuned model

        Args:
            delta_space (float): space of potential delta parameter values
            max_evals (int): maximal number of iterations
            log_model (bool): flag to log the model 
            register_model: flag to register the model
            cv (int): number of cross validation folds by default 3
            idx (list): indices of noisy training data

        Returns:
            log models
        '''
        self.cv = cv
        # Removing noisy data
        if self.noisy_index:
            self.X_train = self.X_train[~self.X_train.index.isin(self.noisy_index)]
            self.train = self.train[~self.train.index.isin(self.noisy_index)]
            self.y_train = self.y_train[~self.y_train.index.isin(self.noisy_index)]
            self.train_pred = self.train_pred[~self.train_pred.index.isin(self.noisy_index)]
            self.train_no_pred = self.train_no_pred[~self.train_no_pred.index.isin(self.noisy_index)]
            self.train_most_sold = self.train_most_sold[~self.train_most_sold.index.isin(self.noisy_index)]
            self.train_least_sold = self.train_least_sold[~self.train_least_sold.index.isin(self.noisy_index)]
            self.filtered = True
        else:
            self._train_test_val_split(test_range_plan_cd=self.test_range_plan_cd)
            self.filtered = False
        # Huber loss-function
        loss_huber = f'Huber:delta={str(delta_space)}'

        # trials = SparkTrials(parallelism=parallelism_value)
        # Hyperopt optimization 
        trials = Trials()
        self.best_delta = fmin(
            fn=self._base_model_objective,
            space=delta_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate = np.random.default_rng(123),
            return_argmin=False,
            verbose=True
        )
        # Best Huber's delta value 
        huber_base_params_fit = self.base_params.copy()
        huber_base_params_fit.update({"loss_function":f'Huber:delta={str(self.best_delta)}'})
        self.huber_base_params_fit = huber_base_params_fit

        # Creating regressor
        reg_huber = ctb.CatBoostRegressor(
           ** self.huber_base_params_fit
        )
        # Calling fucntion to fit and log model
        self.fit_and_log(
            reg=reg_huber,
            log_model=log_model,
            reg_uncert=None,
            register_model=register_model,
        )
        return None
    

    def fit_regressor(
        self,
        delta: float = 9.57,
        params: dict = None,
        n_forecasts: int = 1000,
        iter_size: int = 15,
        log_model: bool = True,
        register_model: bool = False,
        fit_on_all_data: bool = False
    ):
        '''
        Creating model with speicified parameters 

        Args:
            delta (float): delta parameter used in Huber loss-function
            params (dict): parameters for regressor
            n_forecasts (int): number of samples to build in uncertainty model
            iter_size (int): number of virtual ensembles
            log_model (bool): option to log model
            register_model (bool): option to register model
            idx (list): indices of noisy training data
            fit_on_all_data (bool): option to train on all available data

        Returns:
            calling log and fit function
        '''
        self.n_forecasts = n_forecasts
        self.tuned = False
        self.iter_size = iter_size
        
        if fit_on_all_data:
            self.fitted_on_all_data = True
        # removing noisy data
        if self.noisy_index:
            self.X_train = self.X_train[~self.X_train.index.isin(self.noisy_index)]
            self.train = self.train[~self.train.index.isin(self.noisy_index)]
            self.y_train = self.y_train[~self.y_train.index.isin(self.noisy_index)]
            self.train_pred = self.train_pred[~self.train_pred.index.isin(self.noisy_index)]
            self.train_no_pred = self.train_no_pred[~self.train_no_pred.index.isin(self.noisy_index)]
            self.train_most_sold = self.train_most_sold[~self.train_most_sold.index.isin(self.noisy_index)]
            self.train_least_sold = self.train_least_sold[~self.train_least_sold.index.isin(self.noisy_index)]
            self.filtered = True
        else:
            self._train_test_val_split(test_range_plan_cd=self.test_range_plan_cd)
            self.filtered = False
        # Preprocessing of the input parameters
        if params:
            self.tuned = True
            params['iterations']=int(params['iterations'])
            if 'bootstrap' in params:
                bootstrap = params.pop('bootstrap')
                params['bootstrap_type'] = bootstrap['bootstrap_type_c']
                if params['bootstrap_type'] == 'Bayesian':
                    params['bagging_temperature'] = bootstrap['bagging_temperature']
                else:
                    params['subsample'] = bootstrap['subsample']
            # Creating regressor for provided parameters
            reg = ctb.CatBoostRegressor(**self.base_params).set_params(**params)
        # Creating regressor without parameters
        else:
            reg = ctb.CatBoostRegressor(**self.base_params)
        # Calling fit and log function
        self.fit_and_log(
            reg=reg,
            reg_uncert=self.rmse_variance(params=params if params else None),
            log_model=log_model,
            register_model=register_model
        )
        return None
    


    def _objective(self, params):
        '''
        Objective function used in optimization of the model's parameters

        Args:
            params: parameters provided by the Hyperopt iteration
           
        Returns:
            loss: calculated average between rmse and mae 
            status: status of the optimization iteration 
        '''
        with mlflow.start_run(nested=True) as run:
            # preprocessing of provided parameters
            params['iterations']=int(params['iterations'])
            params['bootstrap_type']=params['bootstrap']['bootstrap_type_c']
            if params['bootstrap_type']=='Bayesian':
                params['bagging_temperature']=params['bootstrap']['bagging_temperature']
            else:
                params['subsample']=params['bootstrap']['subsample']
            del params['bootstrap']

            # creating regressor
            reg = ctb.CatBoostRegressor(
                **self.base_params
            ).set_params(**params)
            # wrap custom score function
            model_score = make_scorer(self._scor_func, greater_is_better=False)
            # time series split object
            tcsv = TimeSeriesSplit(n_splits = self.cv)
            # time series cross validation
            loss_dict = cross_validate(
                reg,
                self.X_train,
                np.log(self.y_train) if self.log_y else self.y_train,
                scoring=model_score,
                cv=tcsv,
                return_train_score=False,
            )
            # loss score
            loss = np.mean(loss_dict["test_score"])
        return {"loss": loss, "status": STATUS_OK}

    
    def optimize_and_fit(
        self,
        space_cat: dict,
        delta: float = None,
        max_evals: int = 64,
        #parallelism_value: int = 8,
        iter_size: int = 15,
        n_forecasts: int = 1000,
        log_model: bool = False,
        register_model: bool = False,
        cv: int = 3   
    ):      
        '''
        Optimization of the model's parameters 

        Args:
            space_cat (dict): space of possible parameters
            delta (float): delta parameter can be provide and not optimized
            max_evals (int): maximal number of iterations
            n_forecasts (int): number of samples to build in uncertainty model
            iter_size (int): number of virtual ensembles
            log_model (bool): option to log model
            register_model (bool): option to register model
            remove_noise (bool): option to remove noisy data
            fit_on_all_data (bool): option to train on all available data

        Returns:
            calling fit_and_log function
        '''
        self.tuned = True
        self.cv = cv
        self.iter_size = iter_size
        self.n_forecasts = n_forecasts
        # removing noisy training data
        if self.noisy_index:
            self.X_train = self.X_train[~self.X_train.index.isin(self.noisy_index)]
            self.train = self.train[~self.train.index.isin(self.noisy_index)]
            self.y_train = self.y_train[~self.y_train.index.isin(self.noisy_index)]
            self.train_pred = self.train_pred[~self.train_pred.index.isin(self.noisy_index)]
            self.train_no_pred = self.train_no_pred[~self.train_no_pred.index.isin(self.noisy_index)]
            self.train_most_sold = self.train_most_sold[~self.train_most_sold.index.isin(self.noisy_index)]
            self.train_least_sold = self.train_least_sold[~self.train_least_sold.index.isin(self.noisy_index)]
            self.filtered = True
        else:
            self._train_test_val_split(test_range_plan_cd=self.test_range_plan_cd)
            self.filtered = False
        # Hyperopt optimization
        #trials = SparkTrials(parallelism=parallelism_value)
        trials = Trials()
        self.best_params = fmin(
            fn=self._objective,
            space=space_cat,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate = np.random.default_rng(123),
            return_argmin=False,
            verbose=True)
        # preprocessing of the selected best parameters
        self.best_params['iterations']=int(self.best_params['iterations'])
        self.best_params['bootstrap_type']=self.best_params['bootstrap']['bootstrap_type_c']
        if self.best_params['bootstrap_type']=='Bayesian':
            self.best_params['bagging_temperature']=self.best_params['bootstrap']['bagging_temperature']
        else:
            self.best_params['subsample']=self.best_params['bootstrap']['subsample']
        del self.best_params['bootstrap']
        # fitting regressor
        reg = ctb.CatBoostRegressor(**self.base_params).set_params(**self.best_params)
        # calling function to fit and log
        self.fit_and_log(
            reg=reg,
            reg_uncert=self.rmse_variance(params=self.best_params if self.best_params else None),
            log_model=log_model,
            register_model=register_model)
        return None


    def nll_loss_rmse(self, mean_arr, std_arr):
        '''
        Calculation of a Negative Loglikelihood Function

        Args:
            mean_arr: averages of sample
            std_arr: standard deviations of sample
        
        Returns:
            score: Negative Loglikelihood Value
        '''
        samples = self.sample_from_norm(mean_arr, std_arr)
        nll = []
        for i in range(0, len(mean_arr)):
            mu = np.mean(samples[i,:])
            std = np.std(samples[i,:], ddof = 1)
            nll.append(- np.sum(np.log(st.norm.pdf(samples[i, :],  loc = mu, scale = std))))
        nll = np.asarray(nll)
        score = np.mean(nll)
        return score

    def crps_loss_rmse(self, samples):
        '''
        Calculation of CRPS Function

        Args:
            samples: 2d array, sampled data for each observation
        
        Returns:
            crps: CRPS value
        '''
        crps = []
        mu, std =self.calculate_dist_param(samples)
        for i in range(len(mu)):
            crps.append(ps.crps_gaussian(samples[i,:], mu[i], std[i]))
        crps = np.mean(np.asarray(crps))
        return crps

    def sample_from_norm(self, mean_arr, std_arr):
        '''
        Sample data from Gaussian distribution

        Args:
            mean_arr: averages of sample
            std_arr: standard deviations of sample

        Returns:
            samples: generated samples 
        '''
        samples = []
        for mu, std in zip(mean_arr, std_arr):
            samples.append(st.norm.rvs(loc=mu, scale=std, size=self.n_forecasts, random_state=42))    
        samples = np.asarray(samples)
        return samples

    def calculate_dist_param(self, samples):
        '''
        Calculating mean and standard error from samples sampled from normal distribution

        Args:
            samples: 2d array, sampled data for each observation

        Returns:
            mu: mean
            std: standard error
        '''
        mu = []
        std = []
        for i in range(len(samples[:, 0])):
            mu.append(np.mean(samples[i, :]))
            std.append(np.std(samples[i, :], ddof = 1))
        mu = np.asarray(mu)
        std = np.asarray(std)
        return mu, std

    def rmse_variance(self, params: dict = None):
        '''
        Creatimg model for the uncertainty prediction

        Args:
            params (dict): parameters which should be used in the model

        Returns:
            reg_virt: Unfitted model's regressor
        '''
        self.base_params_uncert = self.base_params.copy()
        self.base_params_uncert.update({
            "loss_function": "RMSEWithUncertainty",
            "posterior_sampling": True,
            "bootstrap_type": 'No'
            })

        if params:
            params['iterations']=int(params['iterations'])
            if 'bootstrap' in params:
                del params['bootstrap']
            reg_virt = ctb.CatBoostRegressor(**self.base_params_uncert).set_params(**params)
        else:
            reg_virt = ctb.CatBoostRegressor(**self.base_params_uncert)

        return reg_virt


    def objective_importance(self,
        params: dict,
        size_sample: float,
        batch_size: int,
        overfit_error: float
    ):
        '''
        Identification of noisy training data

        Args:
            params (dict): parameters which should be used in the model
            size_sample (float): fraction of test data to use in the analysis
            batch_size (int): number of the training data to include to the noisy dataset on each iteration by default 200
            log_transform_target (bool): option to log target variable
        Returns:
            ids_max (list): indices of noisy objects 
        '''
        log_y = self.log_y
        try:
            if self.fitted_on_all_data:
                raise ValueError("Only with test data")
        except:
            pass
        self._train_test_val_split(test_range_plan_cd=self.test_range_plan_cd)
        base_params_object_imp = self.base_params.copy()
        base_params_object_imp.update({
            "loss_function": "RMSE"
        })   
        if params:
            params['iterations']=int(params['iterations'])
            if 'bootstrap' in params:
                bootstrap = params.pop('bootstrap')
                params['bootstrap_type'] = bootstrap['bootstrap_type_c']
                if params['bootstrap_type'] == 'Bayesian':
                    params['bagging_temperature'] = bootstrap['bagging_temperature']
                else:
                    params['subsample'] = bootstrap['subsample']
            reg_init = ctb.CatBoostRegressor(**base_params_object_imp).set_params(**params)
        else:
            reg_init = ctb.CatBoostRegressor(**base_params_object_imp)
        pool_init = ctb.Pool(self.X_train, np.log(self.y_train) if log_y else self.y_train, cat_features=self.cols_index)
        np.random.seed(42)
        test_idx = np.random.choice(list(self.y_test.index), size=size_sample * len(self.y_test), replace=False) if size_sample else list(self.y_test.index)
        pool_sampled_test = ctb.Pool(self.X_test[self.X_test.index.isin(test_idx)], np.log(self.y_test[self.y_test.index.isin(test_idx)]) if log_y else self.y_test[self.y_test.index.isin(test_idx)], cat_features=self.cols_index)
        reg_init.fit(pool_init)
        indices, scores = reg_init.get_object_importance(
            pool_sampled_test,
            pool_init,
            importance_values_sign='Positive',
            update_method='AllPoints'
        )
        indices = pd.DataFrame({"indices": indices, "scores": scores}).sort_values(by='scores', ascending=False)['indices']
        
        rmse_test = np.array([])
        rmse_all = np.array([])
        rmse_train_all = np.array([])
        mae_test = np.array([])
        rmse_train = np.array([])
        mae_train = np.array([])
        ids = []
        metrics_dict = {}
        for batch_end in range(0, len(indices), batch_size):
            idx_iter = list(indices[:batch_end])
            reg_init.fit(self.X_train[~self.X_train.index.isin(idx_iter)], np.log(self.y_train[~self.y_train.index.isin(idx_iter)]) if log_y else self.y_train[~self.y_train.index.isin(idx_iter)])
            preds_all = np.exp(reg_init.predict(self.test[[col for col in self.input_columns]])) if log_y else reg_init.predict(self.test[[col for col in self.input_columns]])
            train_all = np.exp(reg_init.predict(self.X_train[~self.X_train.index.isin(idx_iter)])) if log_y else reg_init.predict(self.X_train[~self.X_train.index.isin(idx_iter)])
            rmse_all = np.append(rmse_all, mean_squared_error(preds_all, self.y_test, squared=False))
            rmse_train_all = np.append(rmse_train_all, mean_squared_error(train_all, self.y_train[~self.y_train.index.isin(idx_iter)], squared=False))
            for country in self.wshop_cd:
                preds_iter = np.exp(reg_init.predict(self.test.query(f'wshop_cd == "{country}"')[[col for col in self.input_columns]])) if log_y else reg_init.predict(self.test.query(f'wshop_cd == "{country}"')[[col for col in self.input_columns]])
                train_iter = np.exp(reg_init.predict(self.train.query(f'wshop_cd == "{country}"')[~self.train.query(f'wshop_cd == "{country}"').index.isin(idx_iter)][[col for col in self.input_columns]])) if log_y else reg_init.predict(self.train.query(f'wshop_cd == "{country}"').query(f'wshop_cd == "{country}"')[~self.train.query(f'wshop_cd == "{country}"').index.isin(idx_iter)][[col for col in self.input_columns]])
                rmse_test = np.append(rmse_test, mean_squared_error(preds_iter, self.test.query(f'wshop_cd == "{country}"')[self.target], squared=False))
                mae_test = np.append(mae_test, mean_absolute_error(preds_iter, self.test.query(f'wshop_cd == "{country}"')[self.target]))
                rmse_train = np.append(rmse_train, mean_squared_error(train_iter, self.train.query(f'wshop_cd == "{country}"')[~self.train.query(f'wshop_cd == "{country}"').index.isin(idx_iter)][self.target], squared=False))
                mae_train = np.append(mae_train, mean_absolute_error(train_iter, self.train.query(f'wshop_cd == "{country}"')[~self.train.query(f'wshop_cd == "{country}"').index.isin(idx_iter)][self.target]))
                ids.append(idx_iter)
                metrics_dict[f'{country}'] = {
                        'rmse_test': rmse_test,
                        'mae_test': mae_test,
                        'rmse_train': rmse_train,
                        'mae_train': mae_train,
                        'index': ids
                }

        min_val = rmse_all[0]
        for  num, val  in enumerate(rmse_all):
                test = val
                train = rmse_train_all[num]
                for country in self.wshop_cd:
                    country_dict = metrics_dict[country]
                    if (test < min_val) & (np.abs((country_dict['rmse_test'][num] - country_dict['rmse_train'][num]) / country_dict['rmse_test'][num]) <= overfit_error):
                        min_val = rmse_all[num]
                        noisy_idx = ids[num]     
        if noisy_idx == ids[0]:
            return None
        else:
            return  noisy_idx



    def fit_and_log(self, reg: Any = None, reg_uncert: Any = None,  log_model: bool = True, register_model: bool = False):                
        '''
        Function for fitting models and log model

        Args:
            reg (Any): regressor of the model for point prediction
            reg_uncert (Any): regressor of the model for uncertainty prediction
            indices (list): indices of removed noisy data from training dataset, default None
            log_model (bool): option to log model by default True
            register_model (bool): option to register model by default False

        Returns:
            None
        '''
        x_train, y_train = (self.X, self.y) if self.fitted_on_all_data else (self.X_train, self.y_train)

        # log input stats
        self._log_input_stats(x_train)

        # fit main model
        reg.fit(x_train, np.log(y_train) if self.log_y else y_train)
        # fit uncertainty model
        reg_uncert.fit(x_train, np.log(y_train) if self.log_y else y_train)
        # log model
        ctb_model = Model_Pyfunc(reg=reg, reg_uncert=reg_uncert, log_y=self.log_y, labels=["good", "ok", "bad"], iter_size=self.iter_size, confidence=self.confidence, n_forecasts=self.n_forecasts, columns=x_train.columns,  target=self.target)
        mlflow.pyfunc.log_model(artifact_path=f"{self.model_name}_model", python_model=ctb_model, registered_model_name=self.model_name if register_model else None)
        # log model params
        mlflow.log_params(reg.get_params())
        # log figures
        self._log_feature_importances(regressor=reg)         
        if not self.fitted_on_all_data:
            self._log_shap_summary(regressor=reg)
            #self._log_prediction_error(regressor=reg)
        mlflow.set_tags(
        {
            "wshop_cd": self.wshop_cd,
            "test_range_plan_cd": self.test_range_plan_cd,
            "tuned": self.tuned,
            "regressor": reg,
            "log_transform_target": self.log_y,
            "fitted_on_all_data": self.fitted_on_all_data,
            "filtered": self.filtered,
            "fit_on_all_countries": self.all_countries
        }
        )

             
        # make predictions and get metrics
        preds = self._make_predictions_new(reg=reg)
        self.metrics = self.update_deep(self.metrics, self._update_metrics_new(predictions=preds))
        self._log_metrics_df()
        if self.fitted_on_all_data == False:
            self._log_dist_metrics_df(self.make_prediction_dist(reg_uncert=reg_uncert))
            try:
                self.log_dist_bias(preds["test"])
            except:
                pass
            if self.noisy_index:
                self._log_ind_noisy_data(self.noisy_index)
        
       
        
        # mlflow does not support nested dict metric tracking
        # flatten metrics dict and log each metric 
        for key, value in self.flatten(self.metrics).items(): 
            mlflow.log_metric(key, value)
        
               
                
                
    def _make_predictions_new(self, reg: Any=None):
        '''
        Function for making predictions on test and train data

        Args:
            reg (Any): regressor of the model for point predictions

        Returns:
            dict_train_test_new (dict): dictionary with predictions
        '''
        self.dict_train_test_new = {}
        for i in self.wshop_cd:

            x_train_wshop = self.X.query(f'wshop_cd == "{i}"') if self.fitted_on_all_data else self.X_train.query(f'wshop_cd == "{i}"')
            self.dict_train_test_new[f'{i}'] = {
                    "train": np.exp(reg.predict(x_train_wshop)) if self.log_y else reg.predict(x_train_wshop),
                    "train_most_sold": np.exp(reg.predict(self.train_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.train_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "train_least_sold": np.exp(reg.predict(self.train_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.train_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "train_predecessor": np.exp(reg.predict(self.train_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.train_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "train_no_predecessor": np.exp(reg.predict(self.train_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.train_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "test": np.exp(reg.predict(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "test_most_sold": np.exp(reg.predict(self.test_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.test_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "test_least_sold": np.exp(reg.predict(self.test_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.test_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "test_no_predecessor": np.exp(reg.predict(self.test_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.test_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]]),
                    "test_predecessor": np.exp(reg.predict(self.test_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])) if self.log_y else reg.predict(self.test_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            }

        return   self.dict_train_test_new 

    def make_prediction_dist(self, reg_uncert: Any = None):
        '''
        Uncertainty predictions if test data exist

        Args:
            reg_uncert (Any): regressor of the model for uncertainty prediction
        
        Returns:
            metrics_df (pd.DataFrame): probabilistics metrics
        '''
        metrics_df = []
        for i in self.wshop_cd:

            virtual = reg_uncert.virtual_ensembles_predict(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], prediction_type='TotalUncertainty', virtual_ensembles_count = self.iter_size)
            virtual_samples = np.exp(self.sample_from_norm(virtual[:,0], np.sqrt(virtual[:,1] + virtual[:,2]))) if self.log_y else self.sample_from_norm(virtual[:,0], np.sqrt(virtual[:,1] + virtual[:,2]))
            virtual_mean, virtual_std = self.calculate_dist_param(virtual_samples) 
            virtual_lower_bound, virtual_upper_bound = st.norm.interval(self.confidence, loc = virtual_mean, scale = virtual_std)
            virtual_confidence_interval = virtual_upper_bound - virtual_lower_bound
            virtual_sharpness = np.mean(virtual_confidence_interval)
            virtual_coverage = np.mean((self.test.query(f'wshop_cd == "{i}"')[self.target] >= virtual_lower_bound) & (self.test.query(f'wshop_cd == "{i}"')[self.target] <= virtual_upper_bound)) * 100
            virtual_nll = self.nll_loss_rmse(virtual_mean, virtual_std)
            virtual_crps = self.crps_loss_rmse(virtual_samples)
            metrics_dict= {
                'virtual_coverage': virtual_coverage,
                'virtual_sharpness': virtual_sharpness,
                'virtual_nll': virtual_nll,
                'virtual_crps': virtual_crps,
                'virtual_mae_test': mean_absolute_error(self.test.query(f'wshop_cd == "{i}"')[self.target], virtual_mean),
                'virtual_rmse_test': mean_squared_error(self.test.query(f'wshop_cd == "{i}"')[self.target], virtual_mean, squared=False),
                'quantile_33': np.quantile(virtual_confidence_interval, 0.33),
                'quantile_66': np.quantile(virtual_confidence_interval, 0.66)
                }
            metrics_df.append(pd.DataFrame([metrics_dict]))
        return metrics_df

        
    def update_deep(self, d, u):
        for k, v in u.items():
            if isinstance(v, coll.Mapping):
                d[k] = self.update_deep(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def flatten(self, dictionary, parent_key=''):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + "." + key if parent_key else key
            if isinstance(value, coll.MutableMapping):
                items.extend(self.flatten(value, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)


    def log_artifact(self, obj, obj_name, file_type, folder = None):
        '''
        Function for logging artifacts as different files

        Args:
            obj: artifact to log
            obj_name: name of the artifact file
            file_type: type of the file
        
        Returns:
            None
        '''
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, obj_name)

            if file_type == 'png':
                mode = 'wb'
                with open(tmp_path, mode) as f:
                    obj.savefig(f, bbox_inches="tight")
            elif file_type == 'csv':
                mode = 'w'
                with open(tmp_path, mode) as f:
                    f.write(obj.to_csv())
            elif file_type == 'html':
                mode = 'w'
                with open(tmp_path, mode) as f:
                    f.write(obj.to_html())
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
            if folder:
                mlflow.log_artifact(tmp_path, artifact_path = folder)
            else:
                mlflow.log_artifact(tmp_path)
        
        
    def _log_metrics_df(self):
        '''
        Log metrics DataFrame as html
        Args:
            None
        Returns:
            None
        '''
        for i in self.wshop_cd:
            metrics_wshop = self.metrics[f'{i}']
            self.log_artifact(pd.DataFrame(metrics_wshop), f'metrics_{i}.html', file_type='html')

    def _log_dist_metrics_df(self, df_list):
        '''
        Logging distribution metrics as html
        Args:
            df (pd.DataFrame): dataframe that contains probabilistic metrics
        Returns:
            None
        '''
        for num, i in enumerate(self.wshop_cd):
            self.log_artifact(df_list[num], f'dist_metrics_{i}.html', file_type='html')

    def _log_ind_noisy_data(self, ind):
        '''
        Logging indices of removed noisy data from the training data

        Args:
            ind: indices of removed rows
        Returns:
            None
        '''
        self.log_artifact(pd.DataFrame(ind), f'indices_noisy_data.html', file_type='html')  
    
    
    def _log_input_stats(self, stats):
        '''
        Logging descriptive statistics of training data

        Args:
            stats (pd.DataFrame):  dataframe to describe
        Returns:
            None
        '''
        self.log_artifact(stats.describe(), f'x_stats.html', file_type='html')
        
        
    def _log_shap_summary(self, regressor):
        '''
        Creating and logging shap charts

        Args:
            regressor: regressor of the model for point prediction
        Returns:
            None
        '''
        explainer=shap.TreeExplainer(regressor)
        for i in self.wshop_cd:

            # All data
            fig = plt.figure(figsize=(12,8))
            shap_values_all=explainer.shap_values(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            shap.summary_plot(shap_values_all, self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], plot_type = 'dot', show=False)
            self.log_artifact(fig, f'shap_all.png', file_type='png', folder=f'shap_{i}')
            plt.show()
            # Predecessor
            fig = plt.figure(figsize=(12,8))
            shap_values_predecessor=explainer.shap_values(self.test_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            shap.summary_plot(shap_values_predecessor, self.test_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], plot_type = 'dot', show=False)
            self.log_artifact(fig, f'shap_predecessor.png', file_type='png', folder=f'shap_{i}')
            plt.show()      
            # No Predecessor
            fig = plt.figure(figsize=(12,8))
            shap_values_no_predecessor=explainer.shap_values(self.test_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            shap.summary_plot(shap_values_no_predecessor, self.test_no_pred.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], plot_type = 'dot', show=False)
            self.log_artifact(fig, f'shap_no_predecessor.png', file_type='png', folder=f'shap_{i}')
            plt.show()  
            # Most Sold
            fig = plt.figure(figsize=(12,8))
            shap_values_most_sold=explainer.shap_values(self.test_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            shap.summary_plot(shap_values_most_sold, self.test_most_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], plot_type = 'dot', show=False)
            self.log_artifact(fig, f'shap_most_sold.png', file_type='png', folder=f'shap_{i}')
            plt.show()  
            # Least Sold
            fig = plt.figure(figsize=(12,8))
            shap_values_least_sold=explainer.shap_values(self.test_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]])
            shap.summary_plot(shap_values_least_sold, self.test_least_sold.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], plot_type = 'dot', show=False)
            self.log_artifact(fig, f'shap_least_sold.png', file_type='png', folder=f'shap_{i}')
            plt.show()  

        
    def _log_feature_importances(self, regressor):
        '''
        Creating and logging feature importances plots

        Args:
            regressor: regressor of the model for point prediction
            x_test (pd.DataFrame): dataset with test data
            y_test (pd.DataFrame): targets of test data
            columns_list (list): list of columns in trainings dataset
        '''
        # PredictionValueChange
        feature_import_prediction_change = pd.DataFrame(sorted(zip(regressor.get_feature_importance(type='PredictionValuesChange'), self.X_train.columns)),
                            columns=["Value", "Feature"])\
                .sort_values("Value", ascending=True)
        fig = plt.figure(figsize=(12,8))
        plt.barh( feature_import_prediction_change['Feature'],  feature_import_prediction_change['Value'])
        plt.title('Feature Importance based on PredictionValuesChange')
        plt.xlabel('Value')
        plt.ylabel('Feature')
        self.log_artifact(obj=fig, obj_name='feature_importance_PredictionValueChange.png', file_type='png', folder='feature_importances_train')
        # Feature statistics 
        feature_stats = regressor.calc_feature_statistics(
            data=self.X_train,
            target=np.log(self.y_train) if self.log_y else self.y_train,
            feature=self.sales_feature,
            plot=False
        )
        fig, ax= plt.subplots(figsize=(32,12))
        labels=['Mean Target','Mean Prediction', 'Prediction for different feature values', 'Number of objects']
        len_graph = range(0,len(feature_stats[self.sales_feature[0]]['mean_target']))
        ax1 = ax.twinx()
        ax.plot(len_graph, np.exp(feature_stats[self.sales_feature[0]]['mean_target']) if self.log_y else feature_stats[self.sales_feature[0]]['mean_target'], marker='*', label=labels[0])

        ax.plot( len_graph, np.exp(feature_stats[self.sales_feature[0]]['mean_prediction']) if self.log_y else feature_stats[self.sales_feature[0]]['mean_prediction'], marker='*', label=labels[1])

        ax.plot( len_graph, np.exp(feature_stats[self.sales_feature[0]]['predictions_on_varying_feature']) if self.log_y else feature_stats[self.sales_feature[0]]['predictions_on_varying_feature'], marker='*', label=labels[2])

        ax1.bar(len_graph, feature_stats[self.sales_feature[0]]['objects_per_bin'], alpha=0.4, color='red', label=labels[3])
        ax.legend(labels=labels)
        ax1.legend(loc='upper center')
        ax.set_ylabel("Prediction and target")
        ax1.set_ylabel("Number of objects")
        ax.set_xlabel("Bins")
        plt.title(f'Statistics for feature {self.sales_feature}')
        self.log_artifact(fig, f'feature_statistics.png', file_type='png', folder='feature_importances_train')
        plt.show()
        
        for i in self.wshop_cd:

            if self.fitted_on_all_data == False:
                # LossFunctionChange
                pool_test = ctb.Pool(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], np.log(self.test.query(f'wshop_cd == "{i}"')[self.target]) if self.log_y else self.test.query(f'wshop_cd == "{i}"')[self.target], cat_features=self.cols_index)
                feature_import_loss_change = pd.DataFrame(sorted(zip(regressor.get_feature_importance(data=pool_test, type='LossFunctionChange'), self.X_train.columns)),
                                    columns=["Value", "Feature"])\
                        .sort_values("Value", ascending=True)
                fig = plt.figure(figsize=(12,8))
                plt.barh( feature_import_loss_change['Feature'],  feature_import_loss_change['Value'])
                plt.title('Feature Importance based on LossFunctionChange')
                plt.xlabel('LossFunctionChange')
                plt.ylabel('Feature')
                self.log_artifact(obj=fig, obj_name=f'feature_importance_LossFunctionChange.png', file_type='png', folder=f'feature_importances_{i}')
                # Interaction
                feature_interaction = pd.DataFrame(regressor.get_feature_importance(data=pool_test, type='Interaction'))
                cols_1 = np.array(self.X_test.columns[np.array(feature_interaction[0], dtype='int')])
                cols_2 = np.array(self.X_test.columns[np.array(feature_interaction[1], dtype='int')])
                cols_comb = cols_1 + ' - ' + cols_2
                interaction_df = pd.DataFrame({'Feature': cols_comb, 'Value': feature_interaction[2]}).sort_values("Value", ascending=True)\
                    .head(20)
                fig = plt.figure(figsize=(12,8))
                plt.barh( interaction_df['Feature'], interaction_df['Value'])
                plt.title('Feature Interaction')
                plt.xlabel('Score')
                plt.ylabel('Feature pairs')
                self.log_artifact(obj=fig, obj_name=f'feature_importance_interaction.png', file_type='png', folder=f'feature_importances_{i}')

    def log_dist_bias(self, prediction):
        for i in self.wshop_cd:
            prediction_wshop = prediction[f'{i}']
            hist_data = [(prediction_wshop - self.test.query(f'wshop_cd == "{i}"')[self.target]) / self.test.query(f'wshop_cd == "{i}"')[self.target], (self.test.query(f'wshop_cd == "{i}"')[self.baseline_qty] - self.test.query(f'wshop_cd == "{i}"')[self.target]) / self.test.query(f'wshop_cd == "{i}"')[self.target]]
            group_labels = ["model_bias", "baseline_bias"]
            fig = ff.create_distplot(hist_data, group_labels, curve_type='kde')
            # combine multiple figures to one graph object
            fig.update_xaxes(title_text="Distribution Deviation in % (Predicted - Real Sales); 1 = 100 %"
                            , showline=True, linewidth=2, linecolor='#0050AA', gridcolor='#0050AA',
                            range = (-4,15)
                            )
            
            fig.update_yaxes(title_text="Density (KDE)", showline=True, linewidth=2, linecolor='#0050AA', gridcolor='#0050AA')
            fig.update_layout(title_text='Distribution of Percentage Errors (Predicted vs. Real Sales)',
                            legend=dict(yanchor="bottom", y=0.8,xanchor="right", x=0.99)
                            , width = 1200
                            , height = 800
                            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, f'distribution_of_bias_{i}.html')
                fig.write_html(tmp_path)
                mlflow.log_artifact(tmp_path)
            fig.show()
    
    #def _log_prediction_error(self, regressor):
    #    '''
    #    Prediction-error plot
    #    Args:
    #        regressor: regressor of the model for point prediction
    #    Returns:
    #        None
    #    '''
    #    # log prediction error plot
    #    
    #    for i in self.wshop_cd:
    #        visualizer = PredictionError(wrap(regressor))
    #        visualizer.score(self.test.query(f'wshop_cd == "{i}"')[[i for i in self.input_columns]], np.log(self.test.query(f'wshop_cd == "{i}"')[self.target]) if self.log_y else self.test.query(f'wshop_cd == "{i}"')[self.target])
#
    #        # Save the figure to a temporary file and log it using the log_artifact function
    #        with tempfile.TemporaryDirectory() as tmp_dir:
    #            tmp_path = os.path.join(tmp_dir, f'prediction_error_{i}.png')
    #            visualizer.show(outpath=tmp_path)
    #            mlflow.log_artifact(tmp_path)
        
        

# COMMAND ----------

class Model_Pyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, reg: Any = None, reg_uncert: Any = None, log_y: bool = False, labels: list = None, iter_size: int = 15, confidence: float = 0.8, n_forecasts: int = 1000, columns: list = None, target: str = None):
        self.log_y = log_y
        self.labels = labels
        self.columns = columns
        self.iter_size = iter_size
        self.n_forecasts = n_forecasts
        self.confidence = confidence
        self.target = target
        self.model = reg
        self.model_uncert = reg_uncert

    def sample_from_norm(self, mean_arr, std_arr):
        samples = []
        for mu, std in zip(mean_arr, std_arr):
            samples.append(st.norm.rvs(loc=mu, scale=std, size=self.n_forecasts))    
        samples = np.asarray(samples)
        return samples

    
    def calculate_dist_param(self, samples):
        mu = []
        std = []
        for i in range(len(samples[:, 0])):
            mu.append(np.mean(samples[i, :]))
            std.append(np.std(samples[i, :], ddof = 1))
        mu = np.asarray(mu)
        std = np.asarray(std)
        return mu, std     
                     

    def get_features(self):
        return {"feature_names": self.columns}
    
    def get_bins(self, bins : dict = None):
        self.bins = bins
        return None

    def predict(self, context, inputs):
        try:
            inputs = inputs[[col for col in self.columns]]
        except:
            raise ValueError("Check Inputs columns")
        point_prediction = np.exp(self.model.predict(inputs)) if self.log_y else self.model.predict(inputs)  
        virtual = self.model_uncert.virtual_ensembles_predict(inputs, prediction_type='TotalUncertainty', virtual_ensembles_count = self.iter_size)
        virtual_samples = np.exp(self.sample_from_norm(virtual[:,0], np.sqrt(virtual[:,1] + virtual[:,2]))) if self.log_y else self.sample_from_norm(virtual[:,0], np.sqrt(virtual[:,1] + virtual[:,2]))
        mean, std = self.calculate_dist_param(virtual_samples) 
        lower_bound, upper_bound = st.norm.interval(self.confidence, loc = mean, scale = std)
        confidence_interval = upper_bound - lower_bound
        lower_bound_pos = np.clip(lower_bound, a_min=0, a_max=None)
        if self.bins:
            bins = self.bins["promo"] if "wt" in self.target else self.bins["after_promo"]
            uncertainty_class = []
            for val in confidence_interval:
                if val < bins[0]:
                    uncertainty_class.append("good")
                elif (val >= bins[0]) & (val < bins[1]):
                    uncertainty_class.append("ok")
                elif val >= bins[1]:
                    uncertainty_class.append("bad")
            uncertainty_class = np.asarray(uncertainty_class)
        prefx = "wt_" if "wt" in self.target  else "nl_"
        return {
            prefx + "point_prediction": point_prediction,
            prefx + "lower_bound": lower_bound_pos,
            prefx + "upper_bound": upper_bound,
            prefx + "confidence_interval": confidence_interval,
            prefx + "uncertainty_class": uncertainty_class if self.bins else None
        }

# COMMAND ----------

def build_inference(frontend, wshop_cd: str = None):
        try:
            frontend = spark.createDataFrame(frontend)
        except:
            pass
        wt_list = [item for sublist in frontend.select(frontend.item_ian_used_wt).collect() for item in sublist][0]
        nl_list = [item for sublist in frontend.select(frontend.item_ian_used_nl).collect() for item in sublist][0]  
        frontend = frontend.withColumn("zipped", arrays_zip(array_union("item_ian_used_wt", "item_ian_used_nl"), array_union("range_plan_cd_wt", "range_plan_cd_nl")))\
            .withColumn("zipped", explode("zipped"))\
            .select("tv_fg", "promotion_medium_type", "sell_off_horizon", "wt_avg_based_on", "nl_avg_based_on", "price", "wshop_cd", "item_ian", "range_plan_cd_predict", col("zipped.0").alias("item_ian_used"), col("zipped.1").alias("range_plan_cd"))\
            .withColumn("wt_flag", when(col("item_ian_used").isin(wt_list), 1)\
                                   .otherwise(0))\
            .withColumn("nl_flag", when(col("item_ian_used").isin(nl_list), 1)\
                                   .otherwise(0))\
            .createOrReplaceTempView("frontend")
        try:
            inference = spark.read.parquet(f'/mnt/paf/data/prediction_sets/aggregated_sales_{wshop_cd}.parquet')
        except:
            raise ValueError("File not found") 
        inference.createOrReplaceTempView("inference")

        return (spark.sql("""
                SELECT
                *,
                /* Decayed sales*/
                CAST(COALESCE( decay_50_wt_avg_sum_sales_hhz,
                               decay_50_wt_avg_sum_sales_ohz,
                               decay_50_wt_avg_sum_sales_ed,
                               decay_50_wt_avg_sum_sales_eu,
                               decay_50_wt_avg_sum_sales_bm) AS INT)
                AS decay_50_wt_avg_sum_sales,
                CAST(COALESCE( decay_25_wt_avg_sum_sales_hhz,
                               decay_25_wt_avg_sum_sales_ohz,
                               decay_25_wt_avg_sum_sales_ed,
                               decay_25_wt_avg_sum_sales_eu,
                               decay_25_wt_avg_sum_sales_bm) AS INT)
                AS decay_25_wt_avg_sum_sales,

                /* number of promotions */
                CAST(COALESCE(num_promo_hhz,
                              num_promo_ohz,
                              num_promo_ed,
                              num_promo_eu,
                              num_promo_bm) AS INT)
                AS wt_pred_num_promo,

                /* promo date */
                CAST(COALESCE(promo_date_hhz,
                              promo_date_ohz,
                              promo_date_ed,
                              promo_date_eu,
                              promo_date_bm) AS INT)
                AS wt_pred_promo_date
                FROM (
                    SELECT 
                    item_front,
                    FIRST_VALUE(range_plan_cd_predict) AS range_plan_cd,
                    FIRST_VALUE(tv_fg) AS tv_fg,
                    FIRST_VALUE(brand_type_cd) AS brand_type_cd,
                    FIRST_VALUE(sell_off_horizon) AS sell_off_horizon,
                    FIRST_VALUE(wt_avg_based_on) AS wt_avg_based_on,
                    FIRST_VALUE(nl_avg_based_on) AS nl_avg_based_on,
                    FIRST_VALUE(promotion_medium_type) AS promotion_medium_type,
                    FIRST_VALUE(price) AS price,
                    COLLECT_SET(item_ian_used) FILTER(WHERE wt_flag =1) AS item_ian_used_wt,
                    COLLECT_SET(item_ian_used) FILTER(WHERE nl_flag =1) AS item_ian_used_nl,
                    MIN(wt_promo_type) AS wt_avg_promo_type,
                    AVG(num_promo) FILTER(WHERE wt_promo_type = 1) AS num_promo_hhz,
                    AVG(num_promo) FILTER(WHERE wt_promo_type = 2) AS num_promo_ohz,
                    AVG(num_promo) FILTER(WHERE wt_promo_type = 3) AS num_promo_ed,
                    AVG(num_promo) FILTER(WHERE wt_promo_type = 4) AS num_promo_eu,
                    AVG(num_promo) FILTER(WHERE wt_promo_type = 5) AS num_promo_bm,
                    MIN(promo_date) FILTER(WHERE wt_promo_type = 1) AS promo_date_hhz,
                    MIN(promo_date) FILTER(WHERE wt_promo_type = 2) AS promo_date_ohz,
                    MIN(promo_date) FILTER(WHERE wt_promo_type = 3) AS promo_date_ed,
                    MIN(promo_date) FILTER(WHERE wt_promo_type = 4) AS promo_date_eu,
                    MIN(promo_date) FILTER(WHERE wt_promo_type = 5) AS promo_date_bm,
                    --AVG(sales_wt) FILTER(WHERE wt_flag =1) AS avg_sales_wt,
                    AVG(sales_nl) FILTER(WHERE nl_flag =1) AS avg_weekly_sales_nl,
                    AVG(daily_sales) AS avg_daily_sales,
                    AVG(daily_sales_nl) FILTER(WHERE nl_flag =1) AS daily_sales_nl,
                    AVG(store_sales_wt) FILTER(WHERE wt_flag =1) AS store_sales_wt,
                    AVG(wt_price) FILTER(WHERE wt_flag =1) AS avg_wt_price,
                    FIRST_VALUE(price) - AVG(wt_price) FILTER(WHERE wt_flag =1) AS wt_diff_in_sales_price,
                    CAST(SUM(POW(0.75, rn_range-1) * nl_price) FILTER(WHERE nl_flag =1) / SUM(POW(0.75, rn_range-1)) FILTER(WHERE wt_flag =1) AS FLOAT) AS decay_75_avg_nl_price,
                    FIRST_VALUE(price) - CAST(SUM(POW(0.75, rn_range-1) * nl_price) FILTER(WHERE nl_flag =1) / SUM(POW(0.75, rn_range-1)) FILTER(WHERE wt_flag =1) AS FLOAT) AS nl_diff_75_in_sales_price,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 1) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 1) AS FLOAT) AS decay_50_wt_avg_sum_sales_hhz,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 2) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 2) AS FLOAT) AS decay_50_wt_avg_sum_sales_ohz,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 3) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 3) AS FLOAT) AS decay_50_wt_avg_sum_sales_ed,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 4) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 4) AS FLOAT) AS decay_50_wt_avg_sum_sales_eu,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 5) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 5) AS FLOAT) AS decay_50_wt_avg_sum_sales_bm,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 1) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 1) AS FLOAT) AS decay_25_wt_avg_sum_sales_hhz,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 2) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 2) AS FLOAT) AS decay_25_wt_avg_sum_sales_ohz,
                    CAST(SUM(POW(0.25, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 3) / SUM(POW(0.25, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 3) AS FLOAT) AS decay_25_wt_avg_sum_sales_ed,
                    CAST(SUM(POW(0.25, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 4) / SUM(POW(0.25, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 4) AS FLOAT) AS decay_25_wt_avg_sum_sales_eu,
                    CAST(SUM(POW(0.25, rn_range-1) * sales_wt) FILTER(WHERE wt_flag =1 AND wt_promo_type = 5) / SUM(POW(0.25, rn_range-1)) FILTER(WHERE wt_flag =1 AND wt_promo_type = 5) AS FLOAT) AS decay_25_wt_avg_sum_sales_bm,
                    CAST(SUM(POW(0.25, rn_range-1) * sales_nl) FILTER(WHERE nl_flag =1) / SUM(POW(0.25, rn_range-1)) FILTER(WHERE nl_flag =1) AS FLOAT) AS decay_25_nl_avg_weekly_sales,
                    CAST(SUM(POW(0.5, rn_range-1) * sales_nl) FILTER(WHERE nl_flag =1) / SUM(POW(0.5, rn_range-1)) FILTER(WHERE nl_flag =1) AS FLOAT) AS decay_50_nl_avg_weekly_sales,
                    CAST(SUM(POW(0.75, rn_range-1) * sales_nl) FILTER(WHERE nl_flag =1) / SUM(POW(0.75, rn_range-1)) FILTER(WHERE nl_flag =1) AS FLOAT) AS decay_75_nl_avg_weekly_sales,
                    CAST(SUM(POW(0.5, rn_range-1) * daily_sales)  / SUM(POW(0.5, rn_range-1))  AS FLOAT) AS decay_50_daily_sales             
                        FROM (
                            SELECT
                            inference.item_ian,
                            inference.range_plan_cd AS range_plan_cd,
                            frontend.item_ian AS item_front,
                            range_plan_cd_predict,
                            sales_wt,
                            sales_nl,
                            daily_sales,
                            daily_sales_nl,
                            store_sales_wt,
                            tv_fg,
                            brand_type_cd,
                            CAST(DECODE(promotion_medium_type, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5)  AS INT) AS promotion_medium_type,
                            sell_off_horizon,
                            wt_avg_based_on,
                            nl_avg_based_on,
                            wt_promo_type,
                            num_promo,
                            promo_date,
                            item_ian_used,
                            nl_price,
                            price,
                            wt_price,
                            wt_flag,
                            nl_flag,
                            DENSE_RANK() OVER(ORDER BY inference.range_plan_cd DESC) AS rn_range
                            FROM inference
                            JOIN frontend
                                 ON frontend.item_ian_used = inference.item_ian
                                AND frontend.range_plan_cd = inference.range_plan_cd
                                AND frontend.wshop_cd = inference.wshop_cd
                        )
                        GROUP BY 1
                 )   
        """)
        ).toPandas() 

# COMMAND ----------

def create_base_set(wshop: str):
    base_set = spark.sql(f"""
    WITH sales_daily AS (
        SELECT
            *
            FROM (
                SELECT
                sales.wshop_cd,
                sales.item_ian,
                sales.range_plan_cd,
                CAST(AVG(order_qty) AS FLOAT) AS daily_sales,
                CAST(AVG(order_qty) FILTER(WHERE sales_type ='nl') AS FLOAT) AS daily_sales_nl
                FROM paf.sales AS sales
                JOIN (
                    SELECT
                    *,
                    DATE_SUB(TO_DATE(CONCAT(CAST(range_plan_cd AS INTEGER), '01'), 'yyMMd'), IF(item_class_cd = 'WDH', 30, 7)) AS cutoff_time
                    FROM ldl.items
                ) AS items
                    ON items.item_ian = sales.item_ian
                    --AND items.range_plan_cd = sales.range_plan_cd
                --WHERE items.cutoff_time >= sales.time_index
                GROUP BY 1, 2, 3
                --HAVING daily_sales != 0
                )
    ),
    nl_prices AS (
        SELECT
            wshop_cd,
            item_ian,
            range_plan_cd,
            MODE(continuous_sales_price) AS nl_price
        FROM paf.sales
        WHERE sales_type = 'nl'
        GROUP BY 1,2,3
    )
    SELECT 
    wshop_cd,
    item_ian,
    range_plan_cd,
    mdm_item_name,
    brand_type_cd,
    wt_promo_type,
    item_name_embeddings_0,
    item_name_embeddings_1,
    item_name_embeddings_2,
    num_promo,
    promo_date,
    sales_wt,
    sales_nl,
    store_sales_wt,
    daily_sales,
    daily_sales_nl,
    wt_price,
    nl_price
    FROM (
        SELECT 
        am_items.wshop_cd,
        am_items.item_ian,
        am_items.range_plan_cd,
        target_wt.target_sales AS sales_wt,
        target_nl.target_sales AS sales_nl,
        am_items.mdm_item_name,
        wt_price,
        nl_price,
        store_sales_wt,
        daily_sales,
        daily_sales_nl,
        item_name_embeddings_0,
        item_name_embeddings_1,
        item_name_embeddings_2,
        CAST(CASE WHEN am_items.brand_type_cd == 'EM' THEN 1 ELSE 0 END AS STRING) AS brand_type_cd,
        DECODE(am_items.sale_promo_type_cd, 'HHZ', 1, 'OHZ', 2, 'EU', 3, 'ED', 4, 'BM', 5) AS wt_promo_type,
        --CAST(COALESCE(wt_avg_pred_sale_promo_type, wt_avg_sim_sale_promo_type) AS STRING) AS wt_avg_promo_type_ian,
        num_promo,
        promo_date
        FROM paf.am_items AS am_items
        JOIN paf.target_wt AS target_wt
            ON am_items.wshop_cd = target_wt.wshop_cd
            AND am_items.item_ian = target_wt.item_ian
            AND am_items.range_plan_cd = target_wt.range_plan_cd
        JOIN ldl.clients AS clients
            ON clients.client_cd = am_items.wshop_cd
        LEFT JOIN paf.target_nl AS target_nl
            ON am_items.item_ian = target_nl.item_ian
            AND am_items.wshop_cd = target_nl.wshop_cd
            AND am_items.range_plan_cd = target_nl.range_plan_cd
        LEFT JOIN (
            SELECT
            clients.client_id,
            mdm_item_sid,
            client_cd,
            store_sales_wt
            FROM (
                SELECT 
                client_id,
                mdm_item_sid,
                CAST(AVG(d7_qty_sales_sum) AS FLOAT) AS store_sales_wt
                FROM ldl.store_item_sales_filtered 
                GROUP BY 1, 2
            ) AS store_sl_agg
            JOIN ldl.clients AS clients
                ON clients.client_id = store_sl_agg.client_id            
        ) AS store_item_sales
            ON store_item_sales.client_cd = am_items.wshop_cd
            AND store_item_sales.mdm_item_sid = am_items.mdm_item_sid
        LEFT JOIN sales_daily
            ON sales_daily.wshop_cd = am_items.wshop_cd
            AND sales_daily.item_ian = am_items.item_ian
            AND sales_daily.range_plan_cd = am_items.range_plan_cd  
        LEFT JOIN nl_prices
            ON nl_prices.item_ian = am_items.item_ian
            AND nl_prices.wshop_cd = am_items.wshop_cd
            AND nl_prices.range_plan_cd = am_items.range_plan_cd
        LEFT JOIN paf_feature_store.semantic_item_name_pca_embeddings AS embeddings
            ON embeddings.mdm_item_sid = am_items.mdm_item_sid
        WHERE target_wt.wshop_cd = '{wshop}'
        ) 

    """).toPandas()
    #os.rmdir(f'/dbfs/mnt/paf/data/prediction_sets/aggregated_sales_{WSHOP_CD}.parquet')
    base_set.to_parquet(f'/dbfs/mnt/paf/data/prediction_sets/aggregated_sales_{wshop}.parquet')
    return base_set
