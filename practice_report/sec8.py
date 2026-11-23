# Section 8 of the report (exec'd by build_report.py)
heading("8 Проверка устойчивости моделей", page_break=True)
para("Высокие метрики на тестовой выборке того же корпуса ещё не означают, что модель будет работать "
     "на других данных. Устойчивость проверялась тремя способами; использовались модели, обученные на "
     "80% корпуса (повтор с seed 42), пороги решения не перенастраивались.")
bullet("**перенос между корпусами**: модели, обученные на SpamAssassin, проверялись на письмах "
       "CSDMC2010, тексты которых не встречаются в SpamAssassin, и наоборот;")
bullet(f"**современный фишинг**: {n_pp} уникальных писем Phishing Pot 2022–2026 гг. Все письма "
       f"вредоносные, поэтому измеряется только доля обнаруженных (recall);")
bullet("**искажение текста**: в тестовом спаме SpamAssassin 10% или 30% слов маскировались (замена букв "
       "похожими символами a→@, o→0, i→1, e→3, s→$ или точки между буквами), либо к письму добавлялись "
       "50 слов, частых в легитимной почте. Легитимные письма не менялись.")
MODELS8 = ["Naive Bayes", "Logistic Regression", "ANN", "CNN", "LSTM", "One-Class SVM (meta)", "DBSCAN (meta)"]
rows = []
for tr_ds, other in [("SA", "CSDMC"), ("CSDMC", "SA")]:
    for m in MODELS8:
        c = e4c[(e4c.trained_on == tr_ds) & (e4c.model == m)].set_index("test_set")
        i = c.loc[f"{tr_ds} test (in-corpus)"]; o = c.loc[f"{other} (unseen texts)"]
        p = c.loc["Phishing Pot 2022-26"]
        rows.append([NAMES[tr_ds], m, pct(i.f1, 1), pct(o.f1, 1), pct(o.recall, 1), pct(o.fpr, 1),
                     pct(p.recall, 1)])
n_other = {tr: int(e4c[(e4c.trained_on == tr) & (e4c.test_set.str.contains("unseen"))].n.iloc[0])
           for tr in ["SA", "CSDMC"]}
table(pd.DataFrame(rows, columns=["Обучение", "Модель", "F1 свой тест, %", "F1 другой корпус, %",
                                  "Recall другой корпус, %", "FPR другой корпус, %",
                                  "Обнаружено Phishing Pot, %"]),
      "Устойчивость моделей на контрольных данных", widths=[2.6, 3.6, 1.9, 2.2, 2.2, 2.1, 2.3], size=9,
      note=f"Другой корпус: CSDMC2010 без писем из SpamAssassin ({n_other['SA']} писем) или SpamAssassin "
           f"без писем из CSDMC2010 ({n_other['CSDMC']} писем). (meta) – обнаружение аномалий по 21 "
           f"индикатору риска.")
for t in SEC8_TRANSFER:
    para(t)
pv = e4p.pivot(index="model", columns="variant", values="recall")
rows = [[m, pct(pv.loc[m, "original"], 1), pct(pv.loc[m, "obf-10"], 1), pct(pv.loc[m, "obf-30"], 1),
         pct(pv.loc[m, "good-words"], 1)] for m in MODELS8]
table(pd.DataFrame(rows, columns=["Модель", "Исходный спам", "Маскировка 10% слов", "Маскировка 30% слов",
                                  "+50 «легитимных» слов"]),
      "Доля обнаруженного спама (recall, %) при искажении текста, SpamAssassin",
      widths=[4.2, 2.6, 3, 3, 3.2],
      note="Модели на метаданных не используют текст, поэтому искажение на них не влияет; они приведены "
           "для сравнения.")
for t in SEC8_PERTURB:
    para(t)
