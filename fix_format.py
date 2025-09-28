content = open("main.py", encoding="utf-8").read()
old_line = "print(f\"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0][\"\"\"lr\"\"\"]:.6f}\")"
new_line = "print(\"Epoch {}, Loss: {:.4f}, LR: {:.6f}\".format(epoch, avg_loss, optimizer.param_groups[0][\"lr\"]))"
content = content.replace(old_line, new_line)
open("main.py", "w", encoding="utf-8").write(content)
print("Fixed with format!")
