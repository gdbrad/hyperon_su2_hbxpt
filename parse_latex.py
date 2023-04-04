import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

import re


import re

def extract_data_(tex_file):
    with open(tex_file, 'r',encoding='utf-8') as f:
        document = f.read()

    # extract baryon data
    baryon_data = {}
    baryon_pattern = r'\\begin{table}{.*?}(.*?)\\end{table}'
    for baryon_table in re.findall(baryon_pattern, document, re.DOTALL):
        name_pattern = r'\\textbf{(.*?)}'
        name = re.search(name_pattern, baryon_table).group(1)
        baryon_data[name] = {}

        # extract E0 values for each baryon
        e0_pattern = r'\$(N_E0.*?)\$'
        for match in re.finditer(e0_pattern, baryon_table):
            e0_name, e0_value = match.group(1), match.group(0)
            baryon_data[name][e0_name] = e0_value

    return baryon_data




def extract_data(tex_doc):
    data = {}
    # extract title
    title_match = re.search(r'\\title\{(.*?)\}', tex_doc, re.DOTALL)
    if title_match:
        data['title'] = title_match.group(1).strip()

    # extract authors
    author_match = re.search(r'\\author\{(.*?)\}', tex_doc, re.DOTALL)
    if author_match:
        authors = author_match.group(1).strip()
        data['authors'] = authors.split('\\and')

    # extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_doc, re.DOTALL)
    if abstract_match:
        data['abstract'] = abstract_match.group(1).strip()

    # extract sections
    section_matches = re.findall(r'\\section\{(.*?)\}', tex_doc, re.DOTALL)
    if section_matches:
        data['sections'] = [match.strip() for match in section_matches]

    return data


def extract_data_(tex_table):
    # Define regular expressions to match the data
    ens_pattern = r"^(a\d+m\d+[^ ]*)"
    e0_pattern = r"{(.+?)}"
    
    # Create a dictionary to store the data
    data = {}
    
    # Parse the table
    for line in tex_table.split("\n"):
        if not line.strip() or line.strip().startswith("%"):
            continue
        if "ens" in line:
            # This is the header row
            continue
        ens_match = re.search(ens_pattern, line)
        if ens_match:
            ens = ens_match.group(1)
            data[ens] = {}
            e0_matches = re.findall(e0_pattern, line)
            data[ens]["N"] = float(e0_matches[0].split()[0])
            data[ens]["Xi"] = float(e0_matches[1].split()[0])
            data[ens]["Sigma"] = float(e0_matches[2].split()[0])
            data[ens]["Lambda"] = float(e0_matches[3].split()[0])
            data[ens]["Xi*"] = float(e0_matches[4].split()[0])
            data[ens]["Delta"] = float(e0_matches[5].split()[0])
            data[ens]["Sigma*"] = float(e0_matches[6].split()[0])
    
    return data


def create_pdf(data):
    # Create a canvas to draw on
    c = canvas.Canvas("e0_comparison.pdf", pagesize=letter)
    
    # Define some formatting options
    font_size = 12
    column_width = 1.5*inch
    row_height = font_size*1.2
    
    # Define the header row
    header = ["Ensemble", "N", "Xi", "Sigma", "Lambda", "Xi*", "Delta", "Sigma*"]
    
    # Write the header row
    x = 0
    y = letter[1] - inch
    for column in header:
        c.drawString(x, y, column)
        x += column_width
    
    # Write the data rows
    y -= row_height
    for ens, values in data.items():
        row = [ens, values["N"], values["Xi"], values["Sigma"], values["Lambda"], values["Xi*"], values["Delta"], values["Sigma*"]]
        x = 0
        for column in row:
            c.drawString(x, y, str(column))
            x += column_width
        y -= row_height
    
    # Save the pdf
    c.save()
