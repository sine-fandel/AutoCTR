from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

doc = SimpleDocTemplate ('Hello.pdf')
styles = getSampleStyleSheet ()
title_style = styles['Title']
heading2_style = styles['Heading2']
code_style = styles['Code']
def_style = styles['Definition']

story = []
story.append (Paragraph ("Data Profile", title_style))

# data schema part 
story.append (Paragraph ("1 DATA SCHEMA", heading2_style))
story.append (Spacer (1, .07 * inch))
story.append (Paragraph ("Note: date schema is shown as the following table to present what data belongs to what types.", def_style))
story.append (Spacer (1, .2 * inch))
data_schema = [['COLUMN NAME', 'FEATURE TYPE'],
				['user', "categorical"],
				['item', "categorical"],
				['ratings', "numerical"]]

t = Table (data_schema, splitByRow=1)
t.setStyle(TableStyle(
    [('BOX', (0, 0), (-1, -1), 1, colors.black),
	('BACKGROUND', (0, 0), (1, 0), colors.lavender),
	("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
	('ALIGN', (0, 0), (-1, -1), 'CENTER')]
    ))
story.append (t)




doc.build (story)

