-- SQLite examples on the local warehouse. Run after `python run_all.py`.

-- Portfolio KPI and monthly default trend.
SELECT strftime('%Y-%m', application_month) AS month,
       COUNT(*) AS application_count,
       SUM(default) AS default_count,
       ROUND(AVG(default), 4) AS default_rate,
       ROUND(AVG(loan_amount), 2) AS avg_loan_amount
FROM dwd_applications
GROUP BY month
ORDER BY month;

-- Risk by acquisition channel; keep the denominator visible.
SELECT channel, COUNT(*) AS application_count,
       SUM(default) AS default_count,
       ROUND(AVG(default), 4) AS default_rate,
       ROUND(AVG(debt_to_income), 4) AS avg_dti
FROM dwd_applications
GROUP BY channel
HAVING COUNT(*) >= 100
ORDER BY default_rate DESC;

-- Lift table for scored applications.
SELECT risk_band, COUNT(*) AS application_count,
       ROUND(AVG(risk_score), 4) AS avg_predicted_risk,
       ROUND(AVG(default), 4) AS observed_synthetic_default_rate
FROM ads_risk_scores
GROUP BY risk_band
ORDER BY CASE risk_band WHEN '高' THEN 1 WHEN '中' THEN 2 ELSE 3 END;

-- High-risk application queue. This is a demo queue, not an approval decision.
SELECT application_id, application_month, region, channel, loan_amount,
       debt_to_income, risk_score, default
FROM ads_risk_scores
WHERE risk_band = '高'
ORDER BY risk_score DESC
LIMIT 100;

