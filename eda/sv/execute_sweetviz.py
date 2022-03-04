import pandas as pd
import sweetviz as sv

for name in ['articles', 'customers', 'sample_submission', 'transactions_train']:
    df = pd.read_csv(f'./input/{name}.csv')
    sv.analyze(df).show_html(filepath=f'./eda/{name}.html', open_browser=False)
