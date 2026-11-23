# Section 9 of the report (exec'd by build_report.py; uses its helpers and loaded results)
heading("9 Обоснованность индикаторов риска и выводы по исследованию", page_break=True)
heading("9.1 Проверка индикаторов риска", 2)
para("Для каждого из 21 индикатора проверено, отличается ли он у спама и легитимных писем. Для "
     "бинарных индикаторов рассчитано отношение шансов (OR) и точный критерий Фишера, для числовых – "
     "AUC (вероятность того, что у спам-письма значение больше, чем у легитимного) и критерий "
     "Манна–Уитни. Порог значимости скорректирован по Бонферрони: 0,05 / 42. Индикатор признавался "
     "**обоснованным**, если в обоих корпусах (SpamAssassin и независимая часть CSDMC2010) он значим, "
     "влияет в одну сторону и даёт заметный эффект (OR ≥ 2 или ≤ 0,5; AUC ≥ 0,6 или ≤ 0,4).")
from features_names import INDICATORS, GROUPS  # noqa: E402
w = e5[e5.set != "PP vs SA ham"].pivot(index="indicator", columns="set", values="effect")
wf = e5.drop_duplicates("indicator").set_index("indicator")
pp = e5[e5.set == "PP vs SA ham"].set_index("indicator")
rows = []
for grp, feats in GROUPS.items():
    for f in feats:
        kind = wf.loc[f, "kind"]
        fmt = (lambda v: num(v, 2)) if kind == "binary" else (lambda v: num(v, 2))
        rows.append([INDICATORS[f], grp, "OR" if kind == "binary" else "AUC", fmt(w.loc[f, "SA"]),
                     fmt(w.loc[f, "CSDMC"]), fmt(pp.loc[f, "effect"]),
                     "да" if wf.loc[f, "well_founded"] else "нет"])
table(pd.DataFrame(rows, columns=["Индикатор", "Группа", "Мера", "SpamAssassin", "CSDMC2010",
                                  "Phishing Pot*", "Обоснован"]),
      "Связь индикаторов риска с угрозой", widths=[5.2, 2.6, 1.3, 2.0, 2.0, 2.0, 1.7], size=9,
      note="OR > 1 и AUC > 0,5 – индикатор чаще или больше у спама/фишинга. * Phishing Pot сравнивается с "
           "легитимными письмами SpamAssassin; столбец показывает направление эффекта для современного "
           "фишинга.")
good = wf[wf.well_founded].index.tolist()
para(f"Обоснованными признаны {len(good)} индикаторов из 21: " +
     ", ".join(INDICATORS[f].lower() for f in good) + ".")
para("Результаты требуют содержательной интерпретации:")
bullet(f"**ссылки.** Ссылка на IP-адрес (OR {num(w.loc['ip_url', 'SA'], 0)} и "
       f"{num(w.loc['ip_url', 'CSDMC'], 0)}) и несовпадение текста ссылки с её адресом (OR "
       f"{num(w.loc['anchor_mismatch', 'SA'], 1)} и {num(w.loc['anchor_mismatch', 'CSDMC'], 1)}) – "
       f"сильные и устойчивые признаки, хотя встречаются лишь в 1–15% вредоносных писем. Плотность "
       f"ссылок, заявленная в плане как индикатор, обоснованной **не** оказалась: на CSDMC2010 она "
       f"выше у спама (AUC {num(w.loc['url_density', 'CSDMC'], 2)}), а на SpamAssassin различия нет "
       f"(AUC {num(w.loc['url_density', 'SA'], 2)}). Сокращённые ссылки в корпусах 2002–2010 гг. "
       f"почти не встречаются, но есть в {pct(pp.loc['shortener_url', 'spam_value'])}% современных "
       f"фишинговых писем;")
bullet(f"**заголовки.** Расхождение домена Message-ID и From чаще у спама (OR "
       f"{num(w.loc['msgid_from_mismatch', 'SA'], 1)} и {num(w.loc['msgid_from_mismatch', 'CSDMC'], 1)}). "
       f"Расхождение From и Return-Path, напротив, характерно для **легитимных** писем (OR "
       f"{num(w.loc['from_returnpath_mismatch', 'SA'], 2)}): списки рассылки переписывают Return-Path. "
       f"Число переходов Received у легитимных писем больше; на CSDMC2010 этот признак почти идеально "
       f"разделяет классы (AUC {num(w.loc['n_received', 'CSDMC'], 2)}), что указывает скорее на "
       f"особенность сбора корпуса, чем на свойство угрозы;")
bullet(f"**вложения.** В этих корпусах вложения чаще у легитимных писем (OR "
       f"{num(w.loc['has_attachment', 'SA'], 2)} и {num(w.loc['has_attachment', 'CSDMC'], 2)}), а опасные "
       f"расширения встречаются слишком редко, чтобы оценить их влияние. Для современного фишинга "
       f"направление обратное (OR {num(pp.loc['has_attachment', 'effect'], 2)}), поэтому вывод о "
       f"вложениях на старых корпусах переносить нельзя;")
bullet(f"**форма содержимого.** Письмо только в HTML (OR {num(w.loc['html_only', 'SA'], 0)} и "
       f"{num(w.loc['html_only', 'CSDMC'], 0)}) и восклицательный знак в теме (OR "
       f"{num(w.loc['subj_exclaim', 'SA'], 1)} и {num(w.loc['subj_exclaim', 'CSDMC'], 1)}) – устойчивые "
       f"признаки спама и современного фишинга.")
rows = [[r.group, int(r.n_indicators), pm(r.auc_sa_test_mean, r.auc_sa_test_std, 3, 1),
         pm(r.auc_csdmc_mean, r.auc_csdmc_std, 3, 1)] for r in e5g.itertuples()]
table(pd.DataFrame(rows, columns=["Группа индикаторов", "Число", "ROC-AUC, тест SpamAssassin",
                                  "ROC-AUC, CSDMC2010 (перенос)"]),
      "Различающая способность групп индикаторов (логистическая регрессия, обучение на SpamAssassin)",
      widths=[5, 1.6, 4.5, 4.5])
ga = e5g.set_index("group")
para(f"Вместе 21 индикатор даёт ROC-AUC {num(ga.loc['Все индикаторы', 'auc_sa_test_mean'])} на тесте "
     f"SpamAssassin и {num(ga.loc['Все индикаторы', 'auc_csdmc_mean'])} при переносе на CSDMC2010 без "
     f"дообучения. Наибольший вклад дают заголовки и форма содержимого; вложения по отдельности "
     f"практически не различают классы (ROC-AUC {num(ga.loc['Вложения', 'auc_sa_test_mean'])}).")

heading("9.2 Выводы по результатам применения ML-методов и графового анализа", 2)
for t in WEEK9_CONCLUSIONS:
    bullet(t)
