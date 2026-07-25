-- cookie_cats_summary.sql
-- Quick descriptive SQL summary of the dataset, runnable against the
-- CSV loaded into SQLite. Included so this project also shows SQL
-- fluency on a different dataset than Project 1, not just Python stats.

-- (Load via: sqlite3 cookie_cats.db
--   .mode csv
--   .import data/cookie_cats.csv players
-- )

SELECT
    version,
    COUNT(*) AS n_players,
    ROUND(AVG(sum_gamerounds), 2) AS avg_rounds,
    ROUND(100.0 * SUM(CASE WHEN retention_1 = 'True' THEN 1 ELSE 0 END) / COUNT(*), 2) AS retention_1_pct,
    ROUND(100.0 * SUM(CASE WHEN retention_7 = 'True' THEN 1 ELSE 0 END) / COUNT(*), 2) AS retention_7_pct
FROM players
GROUP BY version;
