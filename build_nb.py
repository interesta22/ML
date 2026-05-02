import nbformat as nbf
import re

colab_file = 'epl_ann_colab.py'
with open(colab_file, 'r', encoding='utf-8') as f:
    content = f.read()

nb = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell("# ⚽ EPL PLAYER CLASSIFICATION SYSTEM — ANN Model\n## Dynamic Position-Based System | Google Colab Ready\n---"))

pattern = r"(# ─── STEP \d+:.*?\n)(.*?)(?=# ─── STEP \d+:|\Z)"
matches = re.findall(pattern, content, re.DOTALL)

for header, code in matches:
    clean_header = header.replace('# ───', '##').replace('───', '').strip()
    cells.append(nbf.v4.new_markdown_cell(clean_header))
    
    clean_code = code.strip()
    # Uncomment subprocess block for Colab
    if "import subprocess" in clean_code:
        clean_code = clean_code.replace('# import subprocess', 'import subprocess')
        clean_code = clean_code.replace('# subprocess.run', 'subprocess.run')
        clean_code = clean_code.replace('#         "pandas"', '        "pandas"')
    
    if clean_code:
        cells.append(nbf.v4.new_code_cell(clean_code))

nb['cells'] = cells

with open('FPL_ANN_Classification.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook FPL_ANN_Classification.ipynb generated successfully!")
