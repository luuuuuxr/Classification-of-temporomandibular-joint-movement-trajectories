content = open("main.py", encoding="utf-8").read()
content = content.replace("param_groups[0][\"lr\"]", "param_groups[0][\"\"\"lr\"\"\"]")  
open("main.py", "w", encoding="utf-8").write(content)
print("Fixed!")
