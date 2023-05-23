import pandas as pd
from plotnine import *
import sys

if len(sys.argv) != 2:
  print("Error: please provide results csv file.")
  return

results = pd.from_csv(sys.argv[1])
variant_order = ['Dense', 'Specialized', 'SparseRAJA']
results['Variant'] = pd.Categorical(results['Variant'], categories=variant_order)


for benchmark in ['SpMV', 'GauSei', 'InCholFact']
  data = results[results['Benchmark' == benchmark]]
  p = ggplot(data, aes(x='Size',y='Time',fill='Variant'))
  p += facet_wrap('Density',ncol=1)
  p += geom_col(position='dodge')
  p += scale_fill_brewer(type='diverging', palette='PuOr')
  p += theme(axis_title=element_text(size=20),
           axis_text=element_text(size=18),
           legend_text=element_text(size=18),
           legend_title=element_text(size=20),
           text=element_text(size=18)
          )
  outfile_name = benchmark + "_performance.pdf"
  p.save(outfile_name)
