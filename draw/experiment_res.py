import pandas as pd
import matplotlib.pyplot as plt

# 创建一个包含数据的字典
data = {
    "Row": ["A", "B", "C", "D"] * 4,
    "SubRow": [1, 2, 3, 4] * 4,
    "Column1": ["Text1"] * 16,
    "Column2": ["Text2"] * 16,
    "Column3": ["Text3"] * 16,
    "Column4": ["Text4"] * 16,
}

# 将字典转换为 pandas DataFrame
df = pd.DataFrame(data)

# 创建一个透明的画布
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# 创建一个表格并设置数据
table = plt.table(
    cellText=df.iloc[:, 2:].values,
    colLabels=df.columns[2:],
    rowLabels=df["Row"],
    cellLoc="center",
    rowLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)

# 设置单元格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# 添加子行标签
for i, sub_row in enumerate(df["SubRow"]):
    table.add_cell(i + 1, -1, width=0.1, text=sub_row, loc="center")

# 保存表格为图片
plt.savefig("table_with_subrows.png", dpi=300, bbox_inches="tight")

# 显示表格
plt.show()
