import pandas as pd
from plotnine import *
import sys
import scipy
if len(sys.argv) != 2:
  print("Error: please provide results csv file.")
  quit()

results = pd.read_csv(sys.argv[1])

print("Raw data:")
print(results)
results = results.rename(columns=lambda x : x.strip())
results = results.applymap(lambda x: x.strip() if isinstance(x, str) else x)
variant_order = ['Dense', 'Specialized', 'SparseRAJA']
results['Variant'] = pd.Categorical(results['Variant'], categories=variant_order)

results = results.groupby(['Benchmark', 'Size', 'Variant', 'Density']).mean()
print("grouped and meaned:")
print(results)

results = results.reset_index()
print("indices reset:")
print(results)

def charts():
  for benchmark in ['SpMV']:#, 'GauSei', 'InCholFact']:
    for size in [4,6,8]:
      for density in [0.5, 0.1, .001]:
      
        print("Generating figure for", benchmark)
        data = results[results['Benchmark'] == benchmark]
        data = data[data['Size'] == size]
        data = data[data['Density'] == density]
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
        outfile_name = benchmark + "_" + str(size) + "_" + str(density) + "_performance.pdf"
        p.save(outfile_name)

def tables(benchmark):
  data = results[results['Benchmark'] == benchmark]
  data['Time'] = data['Time'] / 10000
  #data['Time'] = data['Time'].astype(int)
  #data['Size'] = data['Size'].astype(str)
  #data['Density'] = data['Density'].astype(str)
  t = ggplot(data, aes(x='factor(Density)',y='factor(Size)'))
  t += aes(fill='Time')
  t += geom_tile(aes(width=0.95, height=0.95))
  t += facet_wrap('Variant')
  t += geom_text(aes(label='Time'), size=9)
  outfile = '_'.join([benchmark, 'table', 'perf.pdf'])
  t.save(outfile)


def lines(benchmark):
  data = results[results['Benchmark'] == benchmark]
  data['Time'] = data['Time'] / 1000
  data['Dim Length'] = data['Size']
  l = ggplot(data, aes(x='Density',y='Time'))
  l += geom_line(aes(color='Variant'))
  l += geom_point(aes(color='Variant',shape='Variant'))
  l += facet_wrap('Dim Length',labeller='label_both')
  l += scale_x_log10()
  l +=  scale_y_log10()
  outfile = '_'.join([benchmark, 'lines', 'perf.pdf'])
  l.save(outfile)


def by_elem_count(benchmark):
  data = results[results['Benchmark'] == benchmark]
  data['Element Count'] = data['Size'] * data['Size'] * data['Density']
  data['Time'] = data['Time'] / 1000
  l = ggplot(data, aes(x='Element Count', y='Time'))
  l += geom_point(aes(color='Variant',shape='factor(Size)'))
  l += scale_x_log10()
  l += scale_y_log10()
  outfile = '_'.join([benchmark, 'ElementCount', 'perf.pdf'])
  l.save(outfile)

def ratio(benchmark):
  data = results[results['Benchmark'] == benchmark]
  pivot = data.pivot_table(index=['Size', 'Density'], columns='Variant', values='Time')
  pivot['Ratio'] = pivot['Specialized'] / pivot['SparseRAJA']
  mean = scipy.stats.gmean(pivot['Ratio'])
  print('\n\n')
  print('Geometric Mean of SparseRAJA speedup relative to Specialized Variant: ', mean)
  print('\n\n')
  
def crossover(benchmark):
  data = results[results['Benchmark'] == benchmark]
  pivot = data.pivot_table(index=['Size', 'Density'], columns='Variant', values='Time')
  pivot['diff'] = pivot['SparseRAJA'] - pivot['Dense']
  print(pivot)


ratio('SpMV')
crossover('SpMV')
quit()
lines('SpMV')
tables('SpMV')
by_elem_count('SpMV')