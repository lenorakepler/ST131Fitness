def cat_display(cat):
	cds = {
		'Amr': 'AMR',
		'Vir': 'Virulence',
		'Stress': 'Stress',
		'Plasmid': 'Plasmid Replicon',
		'Meta': 'Background',
		'nan': 'Background',
		'AMR': 'AMR',
		'VIR': 'Virulence',
		'STRESS': 'Stress',
		'PLASMID': 'Plasmid Replicon',
	}
	return cds[str(cat)]
