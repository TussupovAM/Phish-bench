"""Builds the practice report inside the university template (plan_template.docx).

Every number is read from the result files of the two studies:
  ../phish-bench/results/              phishing URL study (paper)
  ../phish-bench/email_study/results/  e-mail study (weeks 2-9 of the plan)
Output: Отчет_исследовательская_практика_Тусупов.docx
"""
import json
from pathlib import Path

import docx
import pandas as pd
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).parent
U = HERE.parent / "results"
E = HERE.parent / "email_study" / "results"
OUT = HERE / "Отчет_исследовательская_практика_Тусупов.docx"

doc = docx.Document(HERE / "plan_template.docx")
body = doc.element.body
anchor = next(p for p in doc.paragraphs if p.text.strip() == "Отзыв руководителя базы практики")
# the empty paragraph holding the page break right before the review page
anchor = anchor._p.getprevious()


def _move(el):
    anchor.addprevious(el)


def _fmt_run(r, size=12, bold=None, italic=None):
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    r.font.size = Pt(size)
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic


def para(text="", *, align="justify", indent=True, size=12, bold=False, italic=False,
         space_after=4, keep_next=False, page_break_before=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    pf.first_line_indent = Cm(1.25) if indent else Cm(0)
    pf.space_after = Pt(space_after); pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    pf.keep_with_next = keep_next
    pf.page_break_before = page_break_before
    # **bold** segments
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if seg:
            r = p.add_run(seg)
            _fmt_run(r, size, bold=bold or (i % 2 == 1), italic=italic)
    _move(p._p)
    return p


def heading(text, level=1, page_break=False):
    p = para(text, align="left" if level > 1 else "center", indent=False, size=12, bold=True,
             space_after=6, keep_next=True, page_break_before=page_break)
    p.paragraph_format.space_before = Pt(10 if level > 1 else 0)
    return p


def bullet(text):
    p = para("– " + text, indent=False)
    p.paragraph_format.left_indent = Cm(1.25)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    return p


TCOUNT = [0]
FCOUNT = [0]


def table(df, caption, widths=None, size=10, note=None):
    TCOUNT[0] += 1
    para(f"Таблица {TCOUNT[0]} – {caption}", align="left", indent=False, keep_next=True, space_after=2)
    t = doc.add_table(rows=1, cols=df.shape[1])
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c)); _fmt_run(r, size, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            r = cells[j].paragraphs[0].add_run(str(v)); _fmt_run(r, size)
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
    for row in t.rows:
        tr = row._tr
        trPr = tr.get_or_add_trPr()
        cant = trPr.makeelement(qn("w:cantSplit"), {}); trPr.append(cant)
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0); p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing = 1.0
    if widths:
        total = sum(widths)
        widths = [w * 16.5 / total for w in widths]  # fill the text width (16.5 cm)
        t.autofit = False
        tblPr = t._tbl.tblPr
        lay = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"}); tblPr.append(lay)
        for j, w in enumerate(widths):
            t.columns[j].width = Cm(w)
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    # keep the table on one page where possible: every row but the last keeps with the next
    for row in t.rows[:-1]:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True
    _move(t._tbl)
    if note:
        para(note, indent=False, size=10, italic=True, space_after=8)
    else:
        para("", indent=False, size=6, space_after=2)


def figure(path, caption, width_cm=16):
    FCOUNT[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    _move(p._p)
    para(f"Рисунок {FCOUNT[0]} – {caption}", align="center", indent=False, size=11, space_after=8)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}".replace(".", ",")


def num(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def pm(m, s, nd=1, scale=100):
    return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}".replace(".", ",")


def tbl_num(s):
    return s.replace(".", ",") if isinstance(s, str) else s


# =============================================================================
# load results
# =============================================================================
ds = json.load(open(E / "e0_dataset_stats.json"))
env = json.load(open(E / "environment.json"))
e1 = pd.read_csv(E / "e1_anomaly_summary.csv")
e2 = pd.read_csv(E / "e2_summary.csv")
e2t = pd.read_csv(E / "e2_tuning.csv")
e2r = pd.read_csv(E / "e2_runs.csv")
g = json.load(open(E / "e3_graph_stats.json"))
e3p = pd.read_csv(E / "e3_predictive.csv")
e3e = pd.read_csv(E / "e3_ergm.csv", index_col=0)
e4c = pd.read_csv(E / "e4_cross.csv")
e4p = pd.read_csv(E / "e4_perturb.csv")
e5 = pd.read_csv(E / "e5_indicators.csv")
e5g = pd.read_csv(E / "e5_groups.csv")
uci = pd.read_csv(U / "exp1_uci_ml_summary.csv")
ucnn = pd.read_csv(U / "exp2b_uci_cnn_summary.csv")
url = pd.read_csv(U / "exp2_urls_summary.csv")
imb = pd.read_csv(U / "exp3_imbalance.csv")
lat = pd.read_csv(U / "exp4_latency.csv").set_index("model")

ad = ds["after_dedup"]
n_sa = int(ad["0"]["SA"] + ad["1"]["SA"]); n_cs = int(ad["0"]["CSDMC"] + ad["1"]["CSDMC"])
n_pp = int(ad["1"]["PP"])
cs_only = n_cs - ds["csdmc_also_in_sa"]


def e1v(dsn, fs, model, col):
    return e1[(e1.dataset == dsn) & (e1.features == fs) & (e1.model == model)].iloc[0][col]


def e2v(dsn, model, col):
    return e2[(e2.dataset == dsn) & (e2.model == model)].iloc[0][col]


exec(open(HERE / "narrative.py", encoding="utf-8").read())

# =============================================================================
# REPORT BODY
# =============================================================================
heading("СОДЕРЖАНИЕ", page_break=True)
toc = ["Введение",
       "1 Организационная работа и постановка задачи",
       "2 Формирование исследовательской базы данных",
       "3 Разработка моделей обнаружения аномалий",
       "4 Разработка моделей классификации угроз",
       "5 Сравнительный анализ моделей и подбор гиперпараметров",
       "6 Графовый анализ сетевых взаимодействий",
       "7 Применение факторизационных и статистических моделей графа (NMF, ERGM)",
       "8 Проверка устойчивости моделей",
       "9 Обоснованность индикаторов риска и выводы по исследованию",
       "10 Обобщение результатов и подготовка материалов диссертации",
       "Заключение",
       "Список использованных источников",
       "Приложение А. Программная и аппаратная среда, воспроизводимость"]
for t in toc:
    para(t, indent=False, space_after=1)

# ----------------------------------------------------------------------------- intro
heading("ВВЕДЕНИЕ", page_break=True)
para("Исследовательская практика проходила на базе Astana IT University (School of Software "
     "Engineering) в рамках подготовки магистерской диссертации «Разработка методов защиты электронной "
     "почты удалённых объектов от киберугроз». Руководитель практики и научный руководитель – "
     "Орынбек Алибек Сарсембекович.")
para("**Цель практики** – экспериментально проверить методы обнаружения угроз в электронной почте "
     "(спам, фишинг, аномалии трафика) на открытых данных и получить воспроизводимые результаты для "
     "диссертации.")
para("**Задачи практики**, в соответствии с индивидуальным планом-графиком:")
for t in ["сформировать исследовательскую базу данных писем и метаданных (SpamAssassin, UCI ML "
          "Repository, CSDMC2010) и изучить архитектуру email-инфраструктуры;",
          "реализовать методы обнаружения аномалий Isolation Forest, One-Class SVM и DBSCAN;",
          "реализовать и обучить модели классификации ANN, CNN и LSTM, сравнить их по метрикам accuracy, "
          "precision, recall, F1 и подобрать гиперпараметры;",
          "построить граф «отправитель – маршрут – получатель» по метаданным писем и применить модели "
          "NMF и ERGM для поиска структурных аномалий и подозрительных подграфов;",
          "проверить устойчивость моделей на контрольных данных и оценить обоснованность индикаторов "
          "риска (плотность ссылок, аномалии заголовков и вложений);",
          "обобщить результаты и подготовить материалы для диссертации."]:
    bullet(t)
para("Отчёт построен по этапам плана-графика: разделы 1–10 соответствуют неделям 1–10. Все числовые "
     "результаты получены запуском программного кода; код, версии данных и исходные файлы результатов "
     "приведены в открытом репозитории (Приложение А).")

# ----------------------------------------------------------------------------- 1
heading("1 Организационная работа и постановка задачи", page_break=True)
para("На первом этапе изучены база практики и требования к исследовательской практике, составлен "
     "индивидуальный план-график (приведён в начале отчёта) и уточнена задача исследования с опорой на "
     "ранее выполненный обзор литературы по email-угрозам и методам машинного обучения.")
para("Обзор литературы показал три пробела, которые определили постановку задачи:")
bullet("результаты моделей в публикациях получены на разных наборах данных и при разных протоколах, "
       "поэтому их нельзя сравнивать напрямую;")
bullet("модели почти всегда оцениваются на сбалансированных выборках, тогда как в реальном почтовом "
       "трафике доля фишинга мала;")
bullet("метаданные письма (заголовки, маршрут доставки, связи отправителей) используются реже, чем "
       "текст, хотя они доступны до анализа содержимого и труднее подделываются целиком.")
para("Отсюда задача исследования: на открытых данных и по единому протоколу сравнить методы "
     "обнаружения аномалий, классификации и графового анализа для защиты почты удалённых сотрудников, "
     "используя метрики, корректные для несбалансированных данных, и проверить устойчивость моделей "
     "на данных, не использованных при обучении.")

# ----------------------------------------------------------------------------- 2
heading("2 Формирование исследовательской базы данных", page_break=True)
heading("2.1 Наборы данных", 2)
para(f"Исследовательская база включает три корпуса писем с полными заголовками и два набора данных о "
     f"фишинговых веб-ресурсах (таблица 1). Письма разобраны из исходного формата RFC 5322; ошибок "
     f"разбора не было.")
corp = pd.DataFrame([
    ["SpamAssassin public corpus [1]", "письма, 2002 г.", f"{ds['raw_counts']['SA']}",
     f"{ds['dups_within']['SA']}", f"{int(ad['0']['SA'])} / {int(ad['1']['SA'])}", "обучение и тест"],
    ["CSDMC2010 SPAM, TRAINING [2]", "письма, ICONIP 2010", f"{ds['raw_counts']['CSDMC']}",
     f"{ds['dups_within']['CSDMC']}", f"{int(ad['0']['CSDMC'])} / {int(ad['1']['CSDMC'])}", "обучение, тест, контроль"],
    ["Phishing Pot [3] (выборка)", "фишинговые письма, 2022–2026 гг.", f"{ds['raw_counts']['PP']}",
     f"{ds['dups_within']['PP']}", f"0 / {n_pp}", "только контроль"],
    ["UCI Phishing Websites [4]", "30 признаков веб-сайта", "11055", "5206", "2830 / 3019*", "классификация"],
    ["Ebbu2017 [5]", "URL-адреса", "73575", "18941", "29908 / 24726", "классификация"],
], columns=["Набор", "Тип", "Записей", "Дубликатов", "Легит. / вредон. после очистки", "Использование"])
table(corp, "Наборы данных исследовательской базы", widths=[4.2, 3.4, 1.7, 1.9, 3.0, 2.6],
      note="* Для UCI указаны уникальные строки (протокол B). Для Ebbu2017 «дубликаты» включают URL, "
           "совпавшие после удаления схемы http(s)://.")
para(f"При формировании базы обнаружены две проблемы качества данных, которые влияют на оценку моделей:")
bullet(f"**пересечение корпусов.** {ds['csdmc_also_in_sa']} из {n_cs} уникальных писем CSDMC2010 "
       f"({pct(ds['csdmc_also_in_sa'] / n_cs)}%) по тексту совпадают с письмами SpamAssassin. Поэтому при "
       f"перекрёстной проверке (раздел 8) из тестовой выборки исключались письма, встречающиеся в "
       f"обучающем корпусе; независимая часть CSDMC2010 составляет {cs_only} писем;")
bullet("**дубликаты внутри наборов.** 47,1% строк UCI Phishing Websites являются точными копиями других "
       "строк; при перекрёстной проверке копии попадают и в обучение, и в тест и завышают точность.")
para("Метки CSDMC2010 проверены по описанию корпуса: в файле SPAMTrain.label значение 1 означает "
     "легитимное письмо, 0 – спам (2949 и 1378 писем, как указано в readme). Положительным классом во "
     "всех экспериментах считается спам или фишинг.")
heading("2.2 Предобработка", 2)
para("Для каждого письма извлекались: тема и текст (из text/plain, либо из text/html с удалением "
     "разметки), ссылки и их домены, вложения, адреса From, Reply-To, Return-Path, получатели To/Cc, "
     "Message-ID, Date и цепочка заголовков Received. Из цепочки Received определялся первый публичный "
     "IP-адрес – точка входа письма в Интернет. Дубликаты удалялись по хешу нормализованного текста "
     "(тема + тело, нижний регистр, сжатые пробелы).")
para("Для текстовых моделей текст переводился в нижний регистр, а ссылки, адреса и числа заменялись "
     "токенами urltoken, emailtoken и numtoken, чтобы модель не запоминала конкретные адреса. Из "
     "метаданных рассчитан 21 индикатор риска в четырёх группах: ссылки, заголовки, вложения и форма "
     "содержимого (раздел 9).")
heading("2.3 Архитектура email-инфраструктуры и критические точки безопасности", 2)
para("Письмо проходит путь от почтового клиента отправителя (MUA) через сервер отправки (MSA/MTA) и, "
     "возможно, промежуточные ретрансляторы к MX-серверу получателя, затем через фильтр доставки (MDA) "
     "в ящик пользователя. Передача между серверами выполняется по протоколу SMTP [16]; каждый сервер "
     "добавляет в начало письма заголовок Received, поэтому цепочка Received описывает маршрут [17]. "
     "На рисунке 1 отмечены точки, в которых злоумышленник может вмешаться или в которых возможна "
     "проверка.")
figure(E / "fig_smtp_path.png", "Путь письма и критические точки безопасности")
for t in ["(1) Поля From, Reply-To и Return-Path задаёт отправитель, их можно подделать; расхождение "
          "доменов – один из индикаторов риска.",
          "(2) Открытый релей или скомпрометированная учётная запись позволяют отправлять письма от "
          "имени легитимного сервера.",
          "(3) Записи Received добавляет каждый узел; ранние (нижние) записи может сфабриковать "
          "отправитель, надёжны только записи, добавленные доверенными серверами.",
          "(4) MX-сервер получателя проверяет SPF [18], DKIM [19] и DMARC [20] – подлинность IP-адреса "
          "отправителя и подписи домена.",
          "(5) Фильтр анализирует содержимое: ссылки, вложения, текст. Здесь работают модели разделов 3–5.",
          "(6) Последний рубеж – действие пользователя: переход по ссылке или открытие вложения."]:
    bullet(t)
para("Для удалённых сотрудников точки 4–6 особенно важны: письма читаются вне корпоративного периметра, "
     "часто с личных устройств, поэтому фильтрация на уровне почтового сервера и предупреждения "
     "пользователю остаются основной защитой.")

# ----------------------------------------------------------------------------- 3
heading("3 Разработка моделей обнаружения аномалий", page_break=True)
para("Методы обнаружения аномалий не требуют разметки угроз: модель обучается только на легитимных "
     "письмах и помечает как аномалию всё, что на них не похоже. Это полезно для новых видов атак, для "
     "которых ещё нет размеченных примеров.")
para("**Протокол.** Каждый корпус делился стратифицированно 80/20 (три повтора, seeds 42, 43, 44). "
     "Модели обучались на легитимных письмах обучающей части, из которых 20% отложены для выбора "
     "порога: порог равен 95-му процентилю оценки аномальности на этих письмах, то есть целевая доля "
     "ложных срабатываний – 5%. Тестовая часть не использовалась ни для обучения, ни для порога.")
para("**Методы:** Isolation Forest [6] (300 деревьев); One-Class SVM [7] (RBF-ядро, ν = 0,05); "
     "DBSCAN [8] (min_samples = 10, eps – 95-й процентиль расстояния до 10-го соседа). DBSCAN не умеет "
     "оценивать новые точки, поэтому оценкой аномальности служило расстояние до ближайшей ядровой "
     "точки кластеров легитимных писем. Признаки: 21 индикатор риска (meta) и те же индикаторы вместе с "
     "50 компонентами SVD от TF-IDF текста (meta+text).")
rows = []
for dsn in ["SA", "CSDMC"]:
    for fs in ["meta", "meta+text"]:
        for m in ["Isolation Forest", "One-Class SVM", "DBSCAN"]:
            r = e1[(e1.dataset == dsn) & (e1.features == fs) & (e1.model == m)].iloc[0]
            rows.append([f"{'SpamAssassin' if dsn == 'SA' else 'CSDMC2010'}, {fs}", m,
                         num(r.roc_auc_mean), pct(r.precision_mean), pct(r.recall_mean), pct(r.f1_mean),
                         pct(r.fpr_mean)])
table(pd.DataFrame(rows, columns=["Корпус, признаки", "Метод", "ROC-AUC", "Precision, %", "Recall, %",
                                  "F1, %", "FPR, %"]),
      "Обнаружение спама как аномалии (среднее по трём повторам)", widths=[4.2, 3.3, 1.8, 2.0, 1.8, 1.6, 1.6])
best_sa = e1v("SA", "meta", "DBSCAN", "f1_mean"); best_cs = e1v("CSDMC", "meta", "DBSCAN", "f1_mean")
para(f"Лучший результат на обоих корпусах показал DBSCAN на индикаторах риска: ROC-AUC "
     f"{num(e1v('SA', 'meta', 'DBSCAN', 'roc_auc_mean'))} и F1 {pct(best_sa)}% на SpamAssassin, "
     f"ROC-AUC {num(e1v('CSDMC', 'meta', 'DBSCAN', 'roc_auc_mean'))} и F1 {pct(best_cs)}% на CSDMC2010, "
     f"при доле ложных срабатываний около 5%. Легитимные письма образуют несколько плотных кластеров "
     f"(в среднем {num(e1v('SA', 'meta', 'DBSCAN', 'dbscan_clusters'), 1)} на SpamAssassin), а спам лежит "
     f"вне их. One-Class SVM немного уступает (F1 {pct(e1v('SA', 'meta', 'One-Class SVM', 'f1_mean'))}%), "
     f"Isolation Forest при том же пороге находит меньше половины спама "
     f"(recall {pct(e1v('SA', 'meta', 'Isolation Forest', 'recall_mean'))}%).")
para(f"Добавление текстовых признаков ухудшило все методы: у Isolation Forest F1 упал до "
     f"{pct(e1v('SA', 'meta+text', 'Isolation Forest', 'f1_mean'))}%. Легитимные письма разнообразны по "
     f"тематике, поэтому в текстовом пространстве «нормой» становится почти всё. Вывод для системы "
     f"защиты: обнаружение аномалий целесообразно строить на метаданных, а текст анализировать "
     f"классификаторами (раздел 4).")

# ----------------------------------------------------------------------------- 4/5
exec(open(HERE / "sec45.py", encoding="utf-8").read())

# ----------------------------------------------------------------------------- 6
heading("6 Графовый анализ сетевых взаимодействий", page_break=True)
para("Текстовые модели оценивают каждое письмо отдельно. Графовый анализ смотрит на связи: какие "
     "отправители пользуются одной и той же инфраструктурой и кому пишут. Если спам-рассылки "
     "используют общие сети отправки, это видно в структуре графа даже для нового, ранее не "
     "встречавшегося домена отправителя.")
para("**Построение графа.** По метаданным писем SpamAssassin и независимой части CSDMC2010 построен "
     "неориентированный взвешенный граф «отправитель – маршрут – получатель». Вершины трёх типов: домен "
     "отправителя (поле From), сеть /24 точки входа письма в Интернет (первый публичный IP-адрес в "
     "цепочке Received) и домены получателей (To, Cc). Рёбра соединяют вершины вдоль маршрута каждого "
     "письма, вес ребра – число писем. Домены, которыми авторы корпусов заменили реальные адреса при "
     "анонимизации (taint.org, netnoteinc.com, slashnull.org, csmining.org), исключены: они не являются "
     "реальными участниками переписки и превратились бы в искусственные узлы-концентраторы.")
gs = pd.DataFrame([
    ["Писем в графе", f"{g['messages']} (спам: {g['spam']})"],
    ["Писем с известной точкой входа (IP)", f"{pct(g['messages_with_origin_ip'])}%"],
    ["Вершин: домены отправителей / сети /24 / домены получателей",
     f"{g['node_types']['S']} / {g['node_types']['N']} / {g['node_types']['R']}"],
    ["Рёбер", f"{g['edges']}"],
    ["Компонент связности", f"{g['connected_components']}"],
    ["Вершин в наибольшей компоненте", f"{g['largest_component_nodes']} "
     f"({pct(g['largest_component_nodes'] / g['nodes'])}%)"],
], columns=["Характеристика", "Значение"])
table(gs, "Характеристики графа email-взаимодействий", widths=[10, 6])
para(f"Граф разрежен: {g['connected_components']} компонент связности, но "
     f"{pct(g['largest_component_nodes'] / g['nodes'])}% вершин входят в одну большую компоненту, "
     f"связанную через популярные почтовые сервисы и домены получателей. Небольшие отдельные "
     f"компоненты – это, как правило, отправители, которые пользовались собственной инфраструктурой.")
para("**Методика анализа** (NMF и ERGM, раздел 7) описана и подготовлена к согласованию с научным "
     "руководителем; ниже приведены результаты её применения.")

# ----------------------------------------------------------------------------- 7
heading("7 Применение факторизационных и статистических моделей графа (NMF, ERGM)", page_break=True)
heading("7.1 Сообщества по NMF и подозрительные подграфы", 2)
para(f"Для выделения сообществ применена симметричная неотрицательная матричная факторизация "
     f"(SymNMF [12], разновидность NMF [11]): матрица смежности A приближается произведением H·Hᵀ с "
     f"неотрицательной H, и каждая вершина относится к компоненте с наибольшим весом. Факторизация "
     f"применялась к наибольшей компоненте связности, остальные компоненты считались отдельными "
     f"сообществами. Число компонент k выбиралось из {{5, 10, 20, 40, 80}} по модулярности разбиения "
     f"[15] без использования меток; лучшим оказалось k = {g['nmf_k']} (модулярность "
     f"{num(g['nmf_modularity_best'])}). Обычная NMF (A ≈ W·H) объединяла вершины с похожими соседями, "
     f"а не связанные между собой группы, и давала отрицательную модулярность, поэтому от неё "
     f"отказались.")
para(f"Подозрительным подграфом считалось сообщество, в котором не меньше 5 писем и не меньше 90% из "
     f"них – спам. Таких сообществ найдено {g['suspicious_subgraphs']}: {g['suspicious_messages']} писем, "
     f"легитимных писем в них нет ({g['suspicious_ham_in_them']}). Однако они охватывают лишь "
     f"{pct(g['suspicious_spam_captured_share'])}% всего спама. Сообществ с долей спама не выше 10% – "
     f"{g['legit_dominated_subgraphs']}. Крупные сообщества смешанные (рисунок 2).")
figure(E / "fig_graph_communities.png", "Сообщества графа: размер и доля спама", width_cm=15)
pa = e3p.set_index(["score", "subset"])
para(f"Чтобы проверить, несёт ли структура графа полезный сигнал, долю спама в каждом сообществе "
     f"оценивали по обучающей части писем (80/20, три повтора), а для тестовых писем использовали её "
     f"как оценку риска. ROC-AUC составил "
     f"{num(pa.loc[('NMF community spam share', 'all test'), 'roc_auc_mean'])} на всех тестовых письмах и "
     f"{num(pa.loc[('NMF community spam share', 'cold-start senders'), 'roc_auc_mean'])} на письмах от "
     f"доменов, которых не было в обучении. Для сравнения, доля спама по домену отправителя даёт ROC-AUC "
     f"{num(pa.loc[('Sender-domain spam share', 'all test'), 'roc_auc_mean'])} на знакомых доменах, но "
     f"0,5 (случайное угадывание) на новых. Таким образом, граф позволяет оценить риск письма от нового "
     f"домена по тому, через какую инфраструктуру оно отправлено.")
heading("7.2 Экспоненциальная модель случайного графа (ERGM)", 2)
eg = g["ergm"]
para(f"ERGM [14] описывает вероятность наблюдаемого графа через набор структурных статистик и "
     f"позволяет проверить гипотезы о механизмах образования связей. Модель построена для графа "
     f"доменов отправителей: два домена связаны, если письма от них входили в Интернет из одной сети /24 "
     f"({eg['senders']} доменов, из них {eg['spam_senders']} рассылали преимущественно спам; "
     f"{eg['edges']} рёбер). Члены модели: edges (общая плотность), nodefactor (склонность "
     f"спам-отправителей к связям), nodematch (гомофилия: связь между отправителями одного типа) и "
     f"triangle (замыкание треугольников). Параметры оценены методом максимального псевдоправдоподобия "
     f"(MPLE [13]) с выборкой несвязанных пар 1:20 и поправкой свободного члена; стандартные ошибки MPLE "
     f"занижены, поэтому важны знаки и порядок величин, а не точные p-значения.")
er = e3e.rename(index={"edges": "edges", "nodefactor.spam": "nodefactor (спам)",
                       "nodematch.spam": "nodematch (тип отправителя)", "triangle": "triangle"})
table(pd.DataFrame([[i, num(r.coef, 2), num(r.se_mple, 2), num(r.z, 1),
                     ("—" if i == "edges" else num(r.odds_ratio, 2))] for i, r in er.iterrows()],
                   columns=["Член модели", "Коэффициент", "Ст. ошибка (MPLE)", "z", "Отношение шансов"]),
      "Параметры ERGM для графа доменов отправителей", widths=[5, 2.8, 3.2, 1.8, 3.2])
para(f"Главный результат – сильная гомофилия: при прочих равных шансы связи между двумя отправителями "
     f"одного типа (оба спам или оба легитимные) в {num(e3e.loc['nodematch.spam', 'odds_ratio'], 1)} раза "
     f"выше, чем между отправителями разных типов. Из {eg['edges']} рёбер только "
     f"{eg['edge_types']['mixed']} соединяют спам- и легитимного отправителя. Сами по себе "
     f"спам-отправители не связаны чаще других (nodefactor незначим, z = "
     f"{num(e3e.loc['nodefactor.spam', 'z'], 1)}). Положительный коэффициент triangle отражает то, что "
     f"все домены, отправлявшие письма из одной сети, связаны между собой. Вывод: общая "
     f"инфраструктура отправки – содержательный индикатор риска, и его можно использовать для оценки "
     f"новых отправителей.")

# ----------------------------------------------------------------------------- 8
exec(open(HERE / "sec8.py", encoding="utf-8").read())

# ----------------------------------------------------------------------------- 9
exec(open(HERE / "sec9.py", encoding="utf-8").read())

# ----------------------------------------------------------------------------- 10, conclusion, refs
exec(open(HERE / "sec10.py", encoding="utf-8").read())
doc.save(OUT)
print("saved", OUT, "tables", TCOUNT[0], "figures", FCOUNT[0])
