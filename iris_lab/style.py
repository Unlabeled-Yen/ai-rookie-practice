"""圖表共用樣式（類別色取自已驗證的調色盤前三格，散佈圖 all-pairs 皆通過色盲檢查）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
CLASS_COLORS = {"setosa": "#2a78d6", "versicolor": "#eb6834", "virginica": "#1baf7a"}
CLASS_MARKERS = {"setosa": "o", "versicolor": "^", "virginica": "s"}
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_PRIMARY,
    "lines.linewidth": 2,
    "font.size": 10,
})

# 單一色相（藍）連續色階：淺 → 深，用於混淆矩陣熱圖
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
