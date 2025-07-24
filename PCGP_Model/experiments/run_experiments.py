import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PCGP_MODEL import PrincipalComponentGaussianProcessModel
from testfunc_wrapper import TestFuncCaller
from surmise.emulation import emulator
from pyDOE import lhs
import scipy.stats as sps
import pandas as pd
import pathlib

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

rep_n = 1
output_dims = [1, 4, 8, 16]  
ns = [50, 100, 150, 200]
# funcs = ['borehole', 'otlcircuit', 'piston', 'wingweight']  
funcs = ['borehole']  
ntest = 150

def evaluate_model_performance(ytrue, ypred):
    """Calculate RMSE"""
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return rmse

def run_pcgp_model(data, output_dim, n_components=None):
    """Run PCGP model"""
    xtrain, xtest, ytrain = data['xtrain'], data['xtest'], data['ytrain']
    
    pcgp = PrincipalComponentGaussianProcessModel(
        n_components=n_components,
        input_dim=xtrain.shape[1],
        output_dim=output_dim
    )
    
    ranges = np.column_stack([np.zeros(xtrain.shape[1]), np.ones(xtrain.shape[1])])
    
    fitted_model = pcgp.fit(xtrain, ytrain.T, ranges)
    
    pred_mean, _ = fitted_model.predict(xtest, ranges, return_std=True)
    
    return pred_mean.T

def run_surmise_pcgp_model(data, output_dim):
    """Run Surmise PCGP model for comparison"""
    xtrain, xtest, ytrain = data['xtrain'], data['xtest'], data['ytrain']
    
    theta_emu_train = xtrain
    x_emu_train = np.array([[0]])  
    f_emu_train = ytrain.T 
    
    emu = emulator(
        x=x_emu_train,
        theta=theta_emu_train,
        f=f_emu_train,
        method='PCGP',
        options={'epsilon': 0}
    )
    emu.fit()
    
    pred = emu.predict(x=x_emu_train, theta=xtest)
    pred_mean = pred.mean().T  
    
    return pred_mean

def main():
    """Main benchmarking function"""
    np.random.seed(42)
    
    for function in funcs:
        print(f"\n{'='*60}")
        print(f"Processing function: {function}")
        print(f"{'='*60}")
        
        func_caller = TestFuncCaller(function)
        func_meta = func_caller.info
        xdim = func_meta['thetadim']
        locationdim = func_meta['xdim']
        
        xtest = lhs(xdim, ntest)
        
        for output_dim in output_dims:
            print(f"\nOutput dimension: {output_dim}")
            
            locations = sps.uniform.rvs(0, 1, (output_dim, locationdim))
            ytrue_test = func_meta['nofailmodel'](locations, xtest)
            
            generating_noises_var = 0.05 ** ((np.arange(output_dim) + 1) / 2) * np.var(ytrue_test, 1)
            
            for ntrain in ns:
                print(f"  Training size: {ntrain}")
                
                for rep in range(rep_n):
                    print(f"    Repetition: {rep + 1}/{rep_n}")
                    
                    pass

                    # FIXXXXXXX
   
                    data = {
                        'xtrain': xtrain,
                        'xtest': xtest,
                        'ytrain': ytrain,
                        'ytrue': ytrue_test
                    }
 
                    models_to_test = [
                        ('PCGP', run_pcgp_model),
                        ('Surmise_PCGP', run_surmise_pcgp_model)
                    ]
                    
                    for model_name, model_func in models_to_test:
                        try:
                            train_start = time.time()
                            pred_mean = model_func(data, output_dim)
                            train_time = time.time() - train_start
                          
                            rmse = evaluate_model_performance(ytrue_test.T, pred_mean)
           
                            result = {
                                'modelname': model_name,
                                'runno': f'n{ntrain}_rep{rep}',
                                'function': function,
                                'modelrun': f'n{ntrain}_rep{rep}_{function}_{output_dim}',
                                'n': ntrain,
                                'output_dim': output_dim,
                                'rep': rep,
                                'traintime': train_time,
                                'rmse': rmse
                            }
                 
                            df = pd.DataFrame.from_dict(result, orient='index').reset_index()
                            df.columns = ['metric', 'value']
                            output_file = os.path.join(
                                outputdir, 
                                f'{model_name}_{result["modelrun"]}.csv'
                            )
                            df.to_csv(output_file, index=False)
                            
                            print(f"      {model_name}: RMSE={rmse:.4f}, Time={train_time:.2f}s")
                            
                        except Exception as e:
                            print(f"      {model_name}: Failed - {str(e)}")
                            continue

def analyze_results():
    """Analyze and summarize benchmark results"""
    results_files = [f for f in os.listdir(outputdir) if f.endswith('.csv')]
    
    if not results_files:
        print("No results found!")
        return
    
    all_results = []
    
    for file in results_files:
        try:
            df = pd.read_csv(os.path.join(outputdir, file))
        
            parts = file.replace('.csv', '').split('_')
            model_name = parts[0]
  
            result_dict = {'model': model_name}
            for _, row in df.iterrows():
                result_dict[row['metric']] = row['value']
            
            all_results.append(result_dict)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if all_results:
        results_df = pd.DataFrame(all_results)

        summary_file = os.path.join(outputdir, 'benchmark_summary.csv')
        results_df.to_csv(summary_file, index=False)
        
        summary_stats = results_df.groupby(['model', 'function', 'output_dim', 'n']).agg({
            'rmse': ['mean', 'std'],
            'traintime': ['mean', 'std']
        }).round(4)

if __name__ == "__main__":
    main()
    analyze_results()