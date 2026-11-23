# Section 10, conclusion, references and appendix (exec'd by build_report.py)
heading("10 Обобщение результатов и подготовка материалов диссертации", page_break=True)
para("На заключительном этапе результаты обобщены и оформлены в виде материалов для диссертации:")
bullet("подготовлена научная статья «Traditional Machine Learning versus Deep Learning for Phishing "
       "Detection in Remote Work Environments: A Reproducible Comparison of Accuracy, Class Imbalance and "
       "Latency» (результаты раздела 4.2 и 5.3). Статья переработана по замечаниям научного "
       "руководителя: добавлены методология, описание программной и аппаратной среды, проверены все "
       "числовые данные и ссылки, приложен репозиторий с кодом;")
bullet("создан открытый репозиторий Phish-bench с кодом всех экспериментов, скриптами загрузки данных "
       "с фиксированными версиями и файлами исходных результатов, из которых построены таблицы отчёта;")
bullet("подготовлены материалы для глав диссертации: описание исследовательской базы данных и проблем "
       "её качества (раздел 2), результаты обнаружения аномалий (3), классификации и сравнения моделей "
       "(4–5), графового анализа (6–7), проверки устойчивости (8) и обоснованный набор индикаторов "
       "риска (9).")
para("Результаты практики обсуждаются с научным руководителем; замечания руководителя к статье учтены "
     "в её текущей версии.")

heading("ЗАКЛЮЧЕНИЕ", page_break=True)
para("Программа исследовательской практики выполнена: по каждому этапу плана-графика получены "
     "экспериментальные результаты на открытых данных. Основные результаты:")
for t in FINAL_CONCLUSIONS:
    bullet(t)
para("Ограничения работы: корпуса писем SpamAssassin и CSDMC2010 относятся к 2002–2010 гг. и частично "
     "анонимизированы, что влияет на признаки заголовков; современные данные (Phishing Pot) содержат "
     "только фишинг, поэтому на них измерялась лишь доля обнаружения. Гиперпараметры подбирались по "
     "небольшой сетке. Все эксперименты выполнены на одной виртуальной машине без GPU.")
para("Направления дальнейшей работы в рамках диссертации: сбор современного размеченного корпуса "
     "писем с легитимной частью; объединение метаданных, текста и графовых признаков в одной модели; "
     "выбор порога решения под ожидаемую долю угроз; проверка устойчивости к целенаправленным "
     "атакам на текстовые модели.")

heading("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", page_break=True)
refs = [
    "Apache SpamAssassin Project. SpamAssassin public mail corpus. – URL: "
    "https://spamassassin.apache.org/old/publiccorpus/ (использована копия npm-пакета "
    "@stdlib/datasets-spam-assassin 0.2.3).",
    "CSDMC2010 SPAM corpus: data mining competition, ICONIP 2010. – Копия: "
    "https://github.com/hexgnu/spam_filter (коммит f0cfe34).",
    "Peixoto R. F. Phishing Pot: a collection of real phishing samples. – URL: "
    "https://github.com/rf-peixoto/phishing_pot (коммит 78be141).",
    "Mohammad R., McCluskey L. Phishing Websites [Dataset] // UCI Machine Learning Repository. – 2012. – "
    "DOI: 10.24432/C51W2X.",
    "Sahingoz O. K., Buber E., Demir O., Diri B. Machine learning based phishing detection from URLs // "
    "Expert Systems with Applications. – 2019. – Vol. 117. – P. 345–357. – DOI: 10.1016/j.eswa.2018.09.029.",
    "Liu F. T., Ting K. M., Zhou Z.-H. Isolation Forest // Proc. 8th IEEE International Conference on "
    "Data Mining (ICDM). – 2008. – P. 413–422.",
    "Schölkopf B., Platt J. C., Shawe-Taylor J., Smola A. J., Williamson R. C. Estimating the support of a "
    "high-dimensional distribution // Neural Computation. – 2001. – Vol. 13, No. 7. – P. 1443–1471.",
    "Ester M., Kriegel H.-P., Sander J., Xu X. A density-based algorithm for discovering clusters in large "
    "spatial databases with noise // Proc. KDD-96. – 1996. – P. 226–231.",
    "Kim Y. Convolutional neural networks for sentence classification // Proc. EMNLP. – 2014. – "
    "P. 1746–1751.",
    "Hochreiter S., Schmidhuber J. Long short-term memory // Neural Computation. – 1997. – Vol. 9, No. 8. – "
    "P. 1735–1780.",
    "Lee D. D., Seung H. S. Learning the parts of objects by non-negative matrix factorization // Nature. – "
    "1999. – Vol. 401. – P. 788–791.",
    "Kuang D., Ding C., Park H. Symmetric nonnegative matrix factorization for graph clustering // Proc. "
    "SIAM International Conference on Data Mining (SDM). – 2012. – P. 106–117.",
    "Strauss D., Ikeda M. Pseudolikelihood estimation for social networks // Journal of the American "
    "Statistical Association. – 1990. – Vol. 85, No. 409. – P. 204–212.",
    "Robins G., Pattison P., Kalish Y., Lusher D. An introduction to exponential random graph (p*) models "
    "for social networks // Social Networks. – 2007. – Vol. 29, No. 2. – P. 173–191.",
    "Newman M. E. J. Modularity and community structure in networks // PNAS. – 2006. – Vol. 103, No. 23. – "
    "P. 8577–8582.",
    "Klensin J. Simple Mail Transfer Protocol: RFC 5321. – IETF, 2008.",
    "Resnick P. Internet Message Format: RFC 5322. – IETF, 2008.",
    "Kitterman S. Sender Policy Framework (SPF): RFC 7208. – IETF, 2014.",
    "Crocker D., Hansen T., Kucherawy M. DomainKeys Identified Mail (DKIM) Signatures: RFC 6376. – IETF, 2011.",
    "Kucherawy M., Zwicky E. Domain-based Message Authentication, Reporting, and Conformance (DMARC): "
    "RFC 7489. – IETF, 2015.",
    "Shahrivari V., Darabi M. M., Izadi M. Phishing detection using machine learning techniques // "
    "arXiv:2009.11116. – 2020.",
    "Yerima S. Y., Alzaylaee M. K. High accuracy phishing detection based on convolutional neural "
    "networks // arXiv:2004.03960. – 2020.",
    "Pedregosa F. et al. Scikit-learn: machine learning in Python // Journal of Machine Learning Research. – "
    "2011. – Vol. 12. – P. 2825–2830.",
    "Paszke A. et al. PyTorch: an imperative style, high-performance deep learning library // Advances in "
    "Neural Information Processing Systems 32. – 2019.",
    "Tussupov A. Phish-bench: code and results of the experiments. – URL: "
    "https://github.com/TussupovAM/Phish-bench.",
]
for i, r in enumerate(refs, 1):
    p = para(f"{i}. {r}", indent=False, space_after=2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

heading("ПРИЛОЖЕНИЕ А", page_break=True)
heading("Программная и аппаратная среда, воспроизводимость", 2)
para(f"Эксперименты выполнены на виртуальной машине: процессор {env['cpu']} ({env['logical_cpus']} "
     f"логических ядра), {num(env['ram_gb'], 1)} ГБ ОЗУ, без GPU; Linux; Python {env['python']}, NumPy "
     f"{env['numpy']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, PyTorch "
     f"{env['torch'].split('+')[0]} (вычисления на CPU), NetworkX {env['networkx']}, statsmodels "
     f"{env['statsmodels']}. Эксперименты статьи (раздел 4.2), включая замеры задержки, выполнены на другой "
     f"виртуальной машине: Intel Xeon 2,80 ГГц (2 логических ядра), 7,8 ГБ ОЗУ, без GPU, PyTorch 2.14.1, "
     f"XGBoost 3.4.1.")
para("Код, конфигурации и исходные результаты размещены в репозитории "
     "https://github.com/TussupovAM/Phish-bench: каталог email_study – эксперименты с письмами "
     "(разделы 2–9 отчёта), корневой каталог – эксперименты с фишинговыми URL (статья). Скрипт "
     "email_study/scripts/fetch_email_data.sh загружает корпуса фиксированных версий, скрипты "
     "email_study/src/e1_anomaly.py … e5_indicators.py воспроизводят эксперименты, а таблицы отчёта "
     "строятся из файлов email_study/results/*.csv.")
para("Программный код экспериментов и черновик текста отчёта подготовлены с использованием "
     "AI-ассистента Claude (Anthropic). Все числовые результаты получены запуском этого кода и "
     "сверены с файлами результатов.")
