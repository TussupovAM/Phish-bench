# Sections 4 and 5 of the report (exec'd by build_report.py)
NAMES = {"SA": "SpamAssassin", "CSDMC": "CSDMC2010"}
heading("4 Разработка моделей классификации угроз", page_break=True)
heading("4.1 Классификация спама: ANN, CNN, LSTM", 2)
para("В отличие от обнаружения аномалий, классификаторы обучаются на размеченных примерах обоих "
     "классов. Реализованы три нейросетевые модели (PyTorch, вычисления на CPU):")
bullet("**ANN** – многослойный перцептрон на векторе TF-IDF (униграммы и биграммы, 20 000 признаков): "
       "скрытые слои h и 64 нейрона, ReLU, dropout;")
bullet("**CNN** – свёрточная сеть по последовательности слов [9]: эмбеддинг размерности 100, три "
       "параллельные свёртки с окнами 3, 4 и 5 слов, глобальный max-pooling, dropout;")
bullet("**LSTM** – рекуррентная сеть [10]: эмбеддинг 100, слой LSTM (однонаправленный или "
       "двунаправленный), последнее скрытое состояние подаётся на выходной нейрон.")
para("Для CNN и LSTM текст ограничивался первыми 300 словами, словарь – 20 000 слов, встречающихся "
     "хотя бы в двух письмах обучающей выборки. Обучение: Adam (шаг 10⁻³), батч 64, до 15 эпох, "
     "ранняя остановка по функции потерь на валидационной выборке (15% обучающей части). Для "
     "сравнения обучены две классические модели без подбора параметров: наивный байесовский "
     "классификатор и логистическая регрессия на тех же TF-IDF-признаках. Протокол разбиения тот же, "
     "что в разделе 3: 80/20, три повтора; тестовая часть не участвует ни в обучении, ни в подборе "
     "гиперпараметров.")

heading("4.2 Классификация фишинговых URL и веб-сайтов", 2)
para("Для фишинговой части угроз использованы результаты экспериментов, подготовленных для научной "
     "статьи по теме диссертации (код и данные – в том же репозитории). Сравнивались семь классических "
     "моделей (логистическая регрессия, дерево решений, Random Forest, AdaBoost, SVM, KNN, XGBoost) и "
     "нейросетевые модели на двух наборах: UCI Phishing Websites [4] (30 признаков сайта, 10-кратная "
     "кросс-валидация) и Ebbu2017 [5] (54 634 уникальных URL, разбиение 80/20, три повтора).")
rows = []
for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
    a = uci[(uci.model == m) & uci.protocol.str.startswith("A")].iloc[0]
    b = uci[(uci.model == m) & uci.protocol.str.startswith("B")].iloc[0]
    rows.append([m, pct(a.accuracy_mean, 2), pct(b.accuracy_mean, 2), pct(b.f1_mean, 2)])
a = ucnn[ucnn.protocol.str.startswith("A")].iloc[0]; b = ucnn[ucnn.protocol.str.startswith("B")].iloc[0]
rows.append(["1D-CNN (по [22])", pct(a.accuracy_mean, 2), pct(b.accuracy_mean, 2), pct(b.f1_mean, 2)])
table(pd.DataFrame(rows, columns=["Модель", "Accuracy, все строки, %", "Accuracy, без дубликатов, %",
                                  "F1, без дубликатов, %"]),
      "UCI Phishing Websites: влияние дубликатов (10-кратная кросс-валидация)", widths=[5, 3.6, 4, 3.4])
para("На UCI Phishing Websites после удаления дубликатов точность всех моделей снижается (у дерева "
     "решений – на 3,78 п.п.), а нейросеть не превосходит градиентный бустинг: XGBoost – 95,50%, "
     "1D-CNN – 94,31%.")
rows = []
for m in ["Random Forest", "XGBoost", "CNN (char)", "LSTM (char)", "CNN+LSTM (char)"]:
    r = url[url.model == m].iloc[0]
    rows.append([m, pct(r.precision_mean, 2), pct(r.recall_mean, 2), pm(r.f1_mean, r.f1_std, 2),
                 pct(r.fpr_mean, 2), num(lat.loc[m, "latency_median_ms"], 2)])
table(pd.DataFrame(rows, columns=["Модель", "Precision, %", "Recall, %", "F1, %", "FPR, %",
                                  "Задержка, мс/URL"]),
      "Ebbu2017: классификация URL (ML – 20 лексических признаков, DL – символы URL)",
      widths=[4, 2.2, 2, 3, 1.8, 2.8])
para("На сырых URL символьные нейросети заметно лучше: CNN (char) достигает F1 96,84% и в 2,5 раза "
     "меньшей доли ложных срабатываний, чем лучшая классическая модель, при задержке около 0,5 мс на "
     "URL на одном потоке CPU.")

heading("5 Сравнительный анализ моделей и подбор гиперпараметров", page_break=True)
heading("5.1 Подбор гиперпараметров", 2)
para("Гиперпараметры нейросетей подбирались по сетке из четырёх вариантов на каждую модель и корпус "
     "(повтор с seed 42) по F1 на валидационной выборке; при равенстве выбирался вариант с меньшей "
     "функцией потерь.")
rows = []
for dsn in ["SA", "CSDMC"]:
    for m in ["ANN", "CNN", "LSTM"]:
        t = e2t[(e2t.dataset == dsn) & (e2t.model == m)]
        best = t.sort_values(["val_f1", "val_loss"], ascending=[False, True]).iloc[0]
        cfg = json.loads(best.config)
        label = {"hidden": "h", "dropout": "dropout", "filters": "фильтров", "bidirectional": "двунаправл."}
        txt = ", ".join(f"{label[k]} = {('да' if v else 'нет') if isinstance(v, bool) else str(v).replace('.', ',')}"
                        for k, v in cfg.items())
        rows.append([NAMES[dsn], m, f"{pct(t.val_f1.min(), 2)} – {pct(t.val_f1.max(), 2)}", txt,
                     pct(best.val_f1, 2)])
table(pd.DataFrame(rows, columns=["Корпус", "Модель", "F1 валид. по сетке, %", "Выбранная конфигурация",
                                  "F1 валид., %"]),
      "Результаты подбора гиперпараметров", widths=[2.8, 1.6, 3.4, 5.6, 2.2])
spread = e2t.groupby(["dataset", "model"]).val_f1.agg(lambda s: s.max() - s.min())
para(f"Разброс F1 между вариантами сетки невелик – не более {pct(spread.max(), 2)} п.п., то есть "
     f"качество моделей слабо зависит от выбранных гиперпараметров в проверенном диапазоне.")

heading("5.2 Сравнение моделей", 2)
rows = []
order = ["Naive Bayes", "Logistic Regression", "ANN", "CNN", "LSTM"]
for dsn in ["SA", "CSDMC"]:
    for m in order:
        r = e2[(e2.dataset == dsn) & (e2.model == m)].iloc[0]
        rows.append([NAMES[dsn], m, pct(r.accuracy_mean, 2), pct(r.precision_mean, 2), pct(r.recall_mean, 2),
                     pm(r.f1_mean, r.f1_std, 2), pct(r.fpr_mean, 2), num(r.pr_auc_mean, 3)])
table(pd.DataFrame(rows, columns=["Корпус", "Модель", "Accuracy, %", "Precision, %", "Recall, %",
                                  "F1, %", "FPR, %", "PR-AUC"]),
      "Классификация спама: сравнение моделей (тест, среднее ± ст. откл. по трём повторам)",
      widths=[2.6, 3.2, 1.9, 1.9, 1.7, 2.6, 1.5, 1.5], size=9)
for t in SEC5_COMPARISON:
    para(t)

heading("5.3 Метрики при несбалансированных данных", 2)
r1 = imb[(imb.model == "XGBoost") & (imb.target.astype(str) == "0.01")].iloc[0]
r2 = imb[(imb.model == "CNN (char)") & (imb.target.astype(str) == "0.01")].iloc[0]
para(f"Accuracy вводит в заблуждение, когда угроз мало. Для фишинговых URL тестовая выборка была "
     f"пересобрана так, чтобы фишинг составлял 10%, 2% и 1% (все легитимные URL сохранены, фишинговые "
     f"отобраны случайно 50 раз). При 1% фишинга accuracy XGBoost остаётся {pct(r1.accuracy, 1)}%, но "
     f"precision падает до {pct(r1.precision, 1)}%: примерно семь из восьми срабатываний ложные. У CNN "
     f"(char) precision – {pct(r2.precision, 1)}%. Измеренная precision совпадает с расчётной по формуле")
para("Precision(π) = TPR·π / (TPR·π + FPR·(1 − π)),", align="center", indent=False, italic=True)
para("где π – доля угроз, TPR и FPR – доли верных и ложных срабатываний, с точностью до 0,12 п.п. "
     "Поэтому в отчёте, помимо accuracy, приводятся precision, recall, F1, FPR и PR-AUC, а для почтовых "
     "корпусов (доля спама 31–32%) – также результаты на независимых данных (раздел 8).")
