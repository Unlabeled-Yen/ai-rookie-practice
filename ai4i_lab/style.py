"""AI4I 圖表共用樣式（色彩取自已驗證的調色盤；文字一律用中性墨色）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# 類別色，固定順序：slot1 藍、slot2 橘、slot3 綠松、slot4 黃、slot5 洋紅
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 正常 / 故障（散佈圖中最多用前三格，all-pairs 驗證通過）
NORMAL, FAILURE = SERIES[0], SERIES[1]
# 五種故障模式（長條圖相鄰配對驗證通過；淺色格須搭配直接標籤）
MODE_COLORS = {"twf": SERIES[0], "hdf": SERIES[1], "pwf": SERIES[2], "osf": SERIES[3], "rnf": SERIES[4]}

# 連續色階（單一藍色相）與發散色階（藍 ↔ 灰 ↔ 紅）
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f09c9b", "#b8312f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False, "legend.labelcolor": TEXT_PRIMARY, "lines.linewidth": 2, "font.size": 10,
})


def save(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
