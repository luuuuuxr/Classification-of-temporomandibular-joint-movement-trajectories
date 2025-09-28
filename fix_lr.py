import re
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到并替换问题行
content = content.replace(
    'optimizer.param_groups[0][\
\lr\\]',
    \optimizer.param_groups[0][lr]\
)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 修复完成')

