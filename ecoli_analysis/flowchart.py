import schemdraw
import schemdraw.elements as elm
from schemdraw import flow
import matplotlib.pyplot as plt
from transmission_sim.ecoli.analyze import final_dir, writeup_dir

# plt.xkcd()

default_color = "white"

def action_box(label, color=default_color):
	return flow.RoundBox().label(label).fill(color)

def italic(string):
	string = string.replace(' ', '\ ')
	return '\n'.join([r"$\it{" + s + "}$" for s in string.split("\n")])

def bold(string):
	string = string.replace(' ', '\ ')
	return '\n'.join([r"$\bf{" + s + "}$" for s in string.split("\n")])

def software_box(software, description, color=default_color):
	string = bold(software) + "\n" + italic(description)
	return flow.RoundBox().label(string).fill(color)

def software_box_alt(software, description, color=default_color):
	string = bold(description) + "\n" + italic(software)
	return flow.RoundBox().label(string).fill(color)

def description_box(item, description, color=default_color):
	string = bold(item) + "\n" + italic(description)
	return flow.Box().label(string).fill(color)

def draw_ecoli_pipeline():
	d = schemdraw.Drawing()
		
	d.config(fontsize=11)

	software_box = software_box_alt

	down_len = d.unit / 2
	horiz_len = d.unit
	fastq_color = "#35784a"
	alignment_color = "#6453ba"
	bactopia_color = 'lightblue'

	d += (ncbi := flow.Data().label('NCBI SRA'))
	d += flow.Arrow().down(down_len).at(ncbi.S).label('fastq').color(fastq_color)
	d += (qc_reads := software_box('FastQC\nBBTools\nLighter', 'QC +\nclean reads'))

	d += elm.RightLines(arrow='->').delta(-horiz_len, -down_len).at(qc_reads.W).label('cleaned\nfastq\n(short-read)').color(fastq_color)
	d += (snippy := software_box('Snippy', 'SNP calling'))
	
	d += elm.RightLines(arrow='->').delta(horiz_len, -down_len).at(qc_reads.E).label('cleaned\nfastq').color(fastq_color)
	d += (alignment := software_box('Shovill or\nUnicycler', 'de novo\nalignment'))

	d += flow.Arrow().at(alignment.S).label('contigs').color(alignment_color)
	d += (amrfinder := software_box('AMRFinder+\nPlasmidFinder', 'feature\nidentification'))

	d += elm.RightLines(arrow='->').delta(-horiz_len, -2*down_len).at(alignment.W).label('contigs').color(alignment_color)
	d += (cladetyper := software_box('ST131Typer', 'clade typing'))

	d += (bactopia := elm.EncircleBox([qc_reads, alignment], padx=horiz_len/4, pady=down_len/4).fill(bactopia_color).label('In Bactopia', loc='right', rotate=90).zorder(-1))

	d += flow.Arrow().at(snippy.S).down(down_len).label('core SNP\nalignment')
	d += (gubbins := software_box('Gubbins', 'recombination\nmasking'))

	d += flow.Arrow().at(gubbins.S).down(down_len).label('non-recombinant\ncore SNP\nalignment')
	d += (raxml := software_box("RAxML", "maximum-\nlikelihood\nphylogeny\ngeneration"))

	d += flow.Arrow().at(raxml.S).down(down_len).label('maximum-\nlikelihood\nphylogeny')
	d += (lsd := software_box("LSD", "tree dating"))

	d += flow.Arrow().at(lsd.E).right(horiz_len).label('time-\ncalibrated\nphylogeny')
	d += (pastml := software_box("PastML", "ancestral\nreconstruction"))

	d += elm.Arc2(arrow='->', k=-.1).at(amrfinder.S).to(pastml.N).label('Binary\nfeature\npresence/\nabsence', loc="right")

	d.save(final_dir / "base_workflow.png", dpi=300)
	d.save(writeup_dir / "figures" / "base_workflow.png", dpi=300)

def draw_wastewater_pipeline():
	d = schemdraw.Drawing()

	d.config(fontsize=11)

	down_len = d.unit / 2
	horiz_len = d.unit

	fastq_color = "#35784a"
	bam_color = "#6453ba"
	fasta_color = "purple"
	
	bactopia_color = 'lightblue'

	d += (illumina := software_box('Illumina\nBaseSpace', 'Download .fastqs'))
	d += flow.Arrow().down(down_len).at(illumina.S).label('.fastq').color(fastq_color)
	d += (dehost := software_box('Kraken', 'Remove\nhuman reads'))

	d += flow.Arrow().right(horiz_len * 3).at(dehost.E).label('dehosted\n.fastq').color(fastq_color)
	d += (clean := software_box('fastp', 'Fastq\npre-processing\nAnd cleanup').anchor('W'))

	d += flow.Arrow().down(down_len).at(clean.S).label('cleaned\n.fastq').color(fastq_color)
	d += (align := software_box('bwa', 'Align reads to\nSARS-CoV-2\nreference'))

	d += flow.Arrow().down(down_len).at(align.S).label('.bam').color(bam_color)
	d += (trim := software_box('iVar', 'Trim\nprimers'))

	# d += flow.Arrow().down(down_len).at(trim.S).label('.bam').color(bam_color)
	# d += (consensus := software_box('samtools +\niVar', 'Call\nconsensus'))

	d += flow.Arrow().down(down_len).at(trim.S).label('.bam').color(bam_color)
	d += (variants := software_box('samtools +\niVar', 'Call\nvariants'))

	# In QC box
	
	d += flow.Arrow().left(horiz_len).at(variants.W).label('.vcf/.tsv').color(fasta_color)
	d += (freyja := software_box('Freyja', 'Lineage\ndeconvolution'))

	d += elm.RightLines(arrow='->').delta(horiz_len * 1.5, -down_len * 1.7).at(dehost.E).color(fastq_color)
	d += (fastqc := software_box('FastQC', 'Read\nquality\nquantification'))

	# d += elm.RightLines(arrow='->').delta(-horiz_len, -down_len).at(trim.W).label('.bam').color(bam_color)
	d += flow.Arrow().left(horiz_len).at(trim.W).label('.bam').color(bam_color)
	d += (samstats := software_box('samtools', 'Coverage\nquality\nquantification'))

	# Add QC Box
	d += (qc := elm.EncircleBox([freyja, fastqc, samstats], padx=horiz_len/4, pady=down_len/4).fill(bactopia_color).label(bold('QC'), loc='bottom', rotate=0).zorder(-1))
	
	# d += "Predict\nDCIPHER\nQC Score"

	d += flow.Arrow().at(dehost.S).down(down_len).label('dehosted\n.fastq').color(fastq_color)

	# d += flow.Arrow().down(down_len * 3).at(dehost.S).label('dehosted\n.fastq').color(fastq_color)
	d += (qc_pass := flow.Decision(S=bold('YES')).label(bold("Pass QC?")))
	d += flow.Arrow().at(qc_pass.S).down(down_len).label('dehosted\n.fastq').color(fastq_color)
	d += (ncbi := description_box('NCBI', 'Upload to\nBiosample and SRA\ndatabases', color="#dbfcd1"))

	# d += elm.RightLines(arrow='->').delta(-horiz_len, down_len).at(qc.W).label('QC Report').color("gray").linestyle("--")
	d += elm.Arc2(arrow='->', k=-.1).at(qc.W).to(qc_pass.E).label('QC\nReport', loc="left").color("#3128a0")

	d.save("wastewater_workflow.png", dpi=300)

if __name__ == "__main__":
	# draw_wastewater_pipeline()
	draw_ecoli_pipeline()








	# d += flow.Arrow().length(d.unit/2)
	# d += (d2 := flow.Decision(w=5, h=3.9, E='YES', S='NO').label('OKAY,\nYOU SEE THE\nLINE LABELED\n"YES"?'))
	# d += flow.Arrow().length(d.unit/2)
	# d += (d3 := flow.Decision(w=5.2, h=3.9, E='YES', S='NO').label('BUT YOU\nSEE THE ONES\nLABELED "NO".'))

	# d += flow.Arrow().right(d.unit/2).at(d3.E)
	# d += flow.Box(w=2, h=1.25).anchor('W').label('WAIT,\nWHAT?')
	# d += flow.Arrow().down(d.unit/2).at(d3.S)
	# d += (listen := flow.Box(w=2, h=1).label('LISTEN.'))
	# d += flow.Arrow().right(d.unit/2).at(listen.E)
	# d += (hate := flow.Box(w=2, h=1.25).anchor('W').label('I HATE\nYOU.'))

	# d += flow.Arrow().right(d.unit*3.5).at(d1.E)
	# d += (good := flow.Box(w=2, h=1).anchor('W').label('GOOD'))
	# d += flow.Arrow().right(d.unit*1.5).at(d2.E)
	# d += (d4 := flow.Decision(w=5.3, h=4.0, E='YES', S='NO').anchor('W').label('...AND YOU CAN\nSEE THE ONES\nLABELED "NO"?'))

	# d += flow.Wire('-|', arrow='->').at(d4.E).to(good.S)
	# d += flow.Arrow().down(d.unit/2).at(d4.S)
	# d += (d5 := flow.Decision(w=5, h=3.6, E='YES', S='NO').label('BUT YOU\nJUST FOLLOWED\nTHEM TWICE!'))
	# d += flow.Arrow().right().at(d5.E)
	# d += (question := flow.Box(w=3.5, h=1.75).anchor('W').label("(THAT WASN'T\nA QUESTION.)"))
	# d += flow.Wire('n', k=-1, arrow='->').at(d5.S).to(question.S)

	# d += flow.Line().at(good.E).tox(question.S)
	# d += flow.Arrow().down()
	# d += (drink := flow.Box(w=2.5, h=1.5).label("LET'S GO\nDRINK."))
	# d += flow.Arrow().right().at(drink.E).label('6 DRINKS')
	# d += flow.Box(w=3.7, h=2).anchor('W').label('HEY, I SHOULD\nTRY INSTALLING\nFREEBSD!')
	# d += flow.Arrow().up(d.unit*.75).at(question.N)
	# d += (screw := flow.Box(w=2.5, h=1).anchor('S').label('SCREW IT.'))
	# d += flow.Arrow().at(screw.N).toy(drink.S)