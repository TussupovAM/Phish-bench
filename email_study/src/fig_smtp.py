"""Figure: path of an e-mail and the critical security points (week 2), for the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from features import RESULTS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.set_xlim(-1, 101); ax.set_ylim(0, 44); ax.axis("off")

boxes = [  # x, width, label
    (0, 12, "Отправитель\n(MUA)"), (17, 12, "MSA / MTA\nотправителя"), (34, 16, "Промежуточные\nретрансляторы"),
    (55, 12, "MX / MTA\nполучателя"), (72, 11, "Фильтр\n(MDA)"), (88, 12, "Ящик\nсотрудника"),
]
H, Y = 9, 26
for x, w, t in boxes:
    ax.add_patch(FancyBboxPatch((x, Y), w, H, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#eef3fa", ec="#2a5ea8", lw=1))
    ax.text(x + w / 2, Y + H / 2, t, ha="center", va="center")
xs = [x + w for x, w, _ in boxes[:-1]]
xe = [x for x, _, _ in boxes[1:]]
labels = ["SMTP\nsubmission", "SMTP", "SMTP", "", ""]
for a, b, l in zip(xs, xe, labels):
    ax.add_patch(FancyArrowPatch((a + 0.6, Y + H / 2), (b - 0.6, Y + H / 2), arrowstyle="-|>",
                                 mutation_scale=9, color="#333333", lw=1))
    if l:
        ax.text((a + b) / 2, Y + H + 2.2, l, ha="center", va="bottom", fontsize=7, color="#555555")

pts = [  # x centre, number, text
    (6, "1", "Подделка From,\nReply-To, Return-Path"),
    (23, "2", "Открытый релей,\nскомпрометированный\nаккаунт"),
    (42, "3", "Каждый узел добавляет\nReceived; ранние\nзаписи можно подделать"),
    (61, "4", "Проверки SPF, DKIM,\nDMARC по IP источника\nи домену"),
    (77.5, "5", "Ссылки, вложения,\nтекст письма"),
    (94, "6", "Действие\nпользователя"),
]
for x, n, t in pts:
    ax.plot([x, x], [Y - 0.8, 15.5], color="#c0392b", lw=0.8, ls=":")
    ax.add_patch(plt.Circle((x, 13.2), 2.0, color="#c0392b"))
    ax.text(x, 13.2, n, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(x, 10, t, ha="center", va="top", fontsize=6.6)
fig.tight_layout()
fig.savefig(RESULTS / "fig_smtp_path.png", dpi=220)
print("ok")
