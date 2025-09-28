import re

# 修改main.py
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 替换所有出现的类别标签
content = content.replace("[\"Normal\", \"DDwoR\", \"DDwR\"]", "[\"Normal\", \"DDwR\", \"DDwoR\"]")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 已修改main.py中的类别标签顺序")

# 修改model_builder.py
with open("src/models/model_builder.py", "r", encoding="utf-8") as f:
    content = f.read()

# 替换所有出现的类别标签
content = content.replace("[\"Normal\", \"DDwoR\", \"DDwR\"]", "[\"Normal\", \"DDwR\", \"DDwoR\"]")

with open("src/models/model_builder.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 已修改model_builder.py中的类别标签顺序")

