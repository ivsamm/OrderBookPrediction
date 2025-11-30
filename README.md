# Order Book Prediction with XGBoost

## Основные шаги проекта

1. **Предобработка данных**  
   - Сортировка по `local_timestamp`  
   - Создание базовых признаков: `spread`, `mid`, `mid_delta`, суммарная ликвидность `bid_liq`/`ask_liq`, `imbalance`  
   - Разности цен соседних уровней (`bid_price_diff`, `ask_price_diff`)  
   - Доли ликвидности (`bid_qty_ratio`, `ask_qty_ratio`)  
   - Delta-признаки (`spread_delta`, `imbalance_delta`)  
   - Ликвидность топ-5 уровней и соотношение (`liq_top5_ratio`)  
   - Логарифмические признаки для стабильности (`log_bid_liq`, `log_ask_liq`, `log_spread`)  

2. **Создание лагов и роллингов**  
   - Лаги на ключевые признаки: `mid`, `spread`, `bid_liq`, `ask_liq`, `imbalance`  
   - Роллинговые признаки: среднее, стандартное отклонение и нормализация в окне `[5, 10, 20]`  
   - Заполнение NaN после лагов и роллингов нулями  

3. **Балансировка классов**  
   - Вычисление весов классов через `compute_class_weight`  

4. **Обучение модели XGBoost**  
   - Классификатор `XGBClassifier` с оптимальными параметрами:  
     ```python
     subsample=0.7, n_estimators=600, max_depth=8, learning_rate=0.1, colsample_bytree=0.7
     ```  
   - Целевая функция: `multi:softprob` (мультиклассовая задача)  
   - `tree_method="hist"` для ускорения обучения  
   - Контроль качества через `mlogloss` на валидационном наборе  

5. **Оценка модели**  
   - Предсказания на валидационном наборе  
   - Метрика: F1-score с усреднением `macro`
   - Итоговое значение F1-score: 0.37424421387339113


---
