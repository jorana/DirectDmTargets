from common_init import *

# loading normal results.
# from IPython.utils import io
results = {}
all_res = dddm.context['results_dir']
res_dirs = os.listdir(all_res)
no_result = []
load_errors = []
for i, resdir in enumerate(tqdm(res_dirs)):
    try:
        result = dddm.load_multinest_samples_from_file(all_res+'/'+resdir+'/');
    except:
            e = sys.exc_info()[0]
            print(f'Error {e} in loading {resdir}')
            load_errors.append([i, all_res + '/' + resdir, e])
            continue
    if len(result.keys()):
            result['dir'] = all_res+'/'+resdir + '/'
            results[i] = result
    else:
        no_result.append(all_res+'/'+resdir+'/')

def results_to_df(res):
    df = pd.DataFrame()
    items = sorted(list(res.keys()))
    df['item'] = items
    for key in tqdm(res[np.min(list(res.keys()))].keys()):
        if key in ['samples', 'weights', 'weightedsamples']:
            continue
        if key == 'config' or key == 'res_dict':
            for sub_key in res[items[0]][key].keys():
                if sub_key == 'prior':
                    for sub_sub_key in res[items[0]][key][sub_key].keys():
                        if type(res[items[0]][key][sub_key][sub_sub_key]) == dict:
                            for sub_sub_sub_key in res[items[0]][key][sub_key][sub_sub_key].keys():
                                try:
                                    df[key+'_'+sub_key+'_'+sub_sub_key+'_'+sub_sub_sub_key] = [res[it][key][sub_key][sub_sub_key][sub_sub_sub_key] for it in items]
                                except KeyError:
                                    pass
                        else:
                            df[key+'_'+sub_key+'_'+sub_sub_key] = [res[it][key][sub_key][sub_sub_key] for it in items]
                else:
                    try:
                        df[key+'_'+sub_key] = [res[it][key][sub_key] for it in items]
                    except KeyError:
                        pass
            
        else:
            try:
                df[key] = [res[it][key] for it in items]
            except KeyError:
                pass
        
            
    df['mw'] = 10 ** df['config_mw']
    df['n_fit_parameters'] = [len(pars) for pars in df['config_fit_parameters']]
    return df

df = results_to_df(results)


def delete_empty(paths, delete = False, only_old = True):
    for path in tqdm(paths):
        cmd = f"rm -rf {path}"
        if not delete:
            print(cmd)
        if os.path.exists(path) and len(os.listdir(path)) ==0 and delete:
            if only_old:
                t_create = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                dt = datetime.datetime.now() - t_create
                if dt > datetime.timedelta(hours = 100):
                    os.system(cmd)
            else:
                os.system(cmd)

def delete_with_note(df, note, delete = False):
    mask = df.config_notes == note
    paths = df[mask]['dir']
    for path in paths:
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            os.system(cmd)

def delete_with_mask(df, mask, delete = False):
    paths = df[mask]['dir']
    for path in tqdm(paths):
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            os.system(cmd)

def get_posterior(samples, weights):
    # re-scale weights to have a maximum of one
    nweights = weights/np.max(weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    return samples[keepidx,:]

def bin_center(xedges, yedges):
    return 0.5 * (xedges[0:-1] + xedges[1:]), 0.5 * (yedges[0:-1] + yedges[1:])

def get_hist(item, nbins = 45, bin_range = None):
#     nbins = 45
#     bin_range = [[1, 3], [-46, -44]]
    if bin_range == None:
        bin_range = [results[item]['config']['prior']['log_mass']['range'],
                 results[item]['config']['prior']['log_cross_section']['range']
                ]
    counts, xedges, yedges = np.histogram2d(*get_p_i(item), bins = nbins, range = bin_range)
    return counts , xedges, yedges

def get_hist_norm(item):
    counts , xedges, yedges = get_hist(item)
    return counts/np.sum(counts) , xedges, yedges

def get_p_i(i):
#     m, sig = get_posterior(results[i]['samples'], results[i]['weights']).T[:2]
    m, sig = results[i]['weighted_samples'].T[:2]
    
    return np.array([m, sig])

def combine_normalized(items, **plot_kwargs):
    X, Y = np.meshgrid(*get_hist_norm(items[0])[1:])
    for i in items:
        c,_,_ = get_hist_norm(i)
        im = plt.pcolor(X,Y,c.T, norm=LogNorm(vmin = 1e-4,vmax = 1),**plot_kwargs)  
    plt.colorbar()
    
def pow10(x):
    return 10 ** x

def confidence_plot(items, text_box = False, bin_range = None, nsigma = 2, nbins = 50):
    fig,ax=plt.subplots(figsize = (8,6))
    if bin_range == None:
        bin_range = [results[items[0]]['config']['prior']['log_mass']['range'],
                 results[items[0]]['config']['prior']['log_cross_section']['range']
                ]
    
    for k, item in enumerate(items):#, 78, 110 
        x,y =get_p_i(item)
        # Make a 2d normed histogram
        H,xedges,yedges=np.histogram2d(x,y,bins=nbins, range = bin_range, normed=True)
        norm=H.sum() # Find the norm of the sum
        # Set contour levels
        contour3=0.99
        contour2=0.95
        contour1=0.68
        # Take histogram bin membership as proportional to Likelihood
        # This is true when data comes from a Markovian process
        def objective(limit, target):
            w = np.where(H>limit)
            count = H[w]
            return count.sum() - target
        target1 = norm*contour1
        level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
        levels=[level1]
        if nsigma>1:
            target2 = norm*contour2
            level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
            levels.append(level2)
            if nsigma>2:
                target3 = norm*contour3
                level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))    
                levels.append(level3)
            if nsigma>3:
                print('Nsigma too big')
        levels.reverse()
        levels.append(H.max())
        # Find levels by summing histogram to objective
        
        # Pass levels to normed kde plot
        def av_levels(x):
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]

        if levels[0]==levels[1]:
            print("ERRRRRRRRR\n\n")
            print(levels)
            levels[0] /= 1.01
            levels = np.unique(levels)
            print(levels)
        sns_ax = sns.kdeplot(x,y, shade=True,ax=ax,n_levels=levels,cmap="viridis",normed=True, 
                    cbar = False, vmin=levels[0], vmax=levels[-1])
        kwargs = {}
        if k == 0:
            kwargs['label'] = 'best fit'
        plt.scatter(np.mean(x),np.mean(y), c='black',
                    marker = '+',**kwargs)
        if k == 0:
            kwargs['label'] = 'benchmark value'
        plt.scatter(results[item]['config']['mw'],
                    results[item]['config']['sigma'], c='blue',
                    marker = 'x',
                    **kwargs)
        if k == 0:
            cbar = ax.figure.colorbar(sns_ax.collections[0])
            cbar.set_ticks(av_levels(np.linspace(0,1,nsigma+1)))
            col_labels = ['$3\sigma$', '$2\sigma$', '$1\sigma$'][3-nsigma:]
            cbar.set_ticklabels(col_labels)
            cbar.set_label("Posterior probability")
    secax = ax.secondary_xaxis('top', functions=(pow10, np.log10))
    if 'migd' in results[items[0]]['config']['detector']:
        x_ticks = [0.01, 0.1, 1, 3]
    else:
        x_ticks = [15, 25, 50, 100, 250, 500, 1000]
    for x_tick in x_ticks:
        ax.axvline(np.log10(x_tick), alpha = 0.1)
    secax.set_ticks(x_ticks)
    plt.xlim(np.log10(x_ticks[0]),np.log10(x_ticks[-1]))
    plt.xlabel("$\log_{10}(M_{\chi}$ $[GeV/c^{2}]$)")
    secax.set_xlabel("$M_{\chi}$ $[GeV/c^{2}]$")
    plt.ylabel("$\log_{10}(\sigma_{S.I.}$ $[cm^{2}]$)")
    plt.legend(loc = 'upper right')

    if text_box:
        plt.text(0.05, 0.95, text_box, 
                 bbox=dict(facecolor="white",
                           boxstyle="round"), 
                 transform=ax.transAxes,
                 alpha=0.5)
        
def find_largest_posterior(df, sig = -38, mw = 1, fix_nlive= None):
    items = []
    for nparam in [2,5]:
        for halo in ['shm', 'shielded_shm']:
            mask = (
                (df['n_fit_parameters'] == nparam) & 
                (df['config_halo_model'] == halo) &
                (df['mw'] == mw) &
                (df['config_sigma'] == sig))
            if fix_nlive:
                mask = mask & (df['config_nlive'] == fix_nlive)
            sel_df = df[mask].sort_values('config_nlive')
            
            maskA = sel_df['config_halo_model'] == 'shielded_shm'
            maskB = np.array(['VerneSHM' in model for model in sel_df['config_halo_model'].values])
            mask2 = maskA | maskB
            print(np.sum(maskA), np.sum(maskB), np.sum(mask2))
            if halo == 'shielded_shm':
                sel_df = sel_df[mask2]
            else:
                sel_df = sel_df[~mask2]
            print(f'{halo} for {nparam} pars @ s = {sig}, m = {mw}')
            if len(sel_df):
                print(sel_df[['item', 'mw', 'config_sigma', 'config_nlive']][-2:-1])
                items.append(sel_df['item'].values[-1])
    return items
    
det = 'Xe_migd'
save_dir = '2020_02_24_sanity_checks/'
def overlay_hist_confidence_info(i, save_label = '', bin_range = None):
    note = results[i]['config']['notes']
    this_df = df[df['item'] == i]
#     print(i, results[i]['config']['notes'],results[i]['config']['prior'])
    print(this_df[['item', 'mw', 'config_sigma', 'n_fit_parameters', 'config_halo_model', 'config_nlive']])
    
    
    bin_range = [results[i]['config']['prior']['log_mass']['range'], 
        results[i]['config']['prior']['log_cross_section']['range']] if bin_range == None else bin_range
    confidence_plot([i], text_box = f'{det}-detector', nsigma = 2, nbins = 50,
                   bin_range = bin_range)
    title = f'$m_w$={this_df["mw"].values[0]}'+' $Gev/c^{2}$'
    title += f', $\log(\sigma)$ = {this_df["config_sigma"].values[0]}\n'
    title += f'model ={this_df["config_halo_model"].values[0]}'
    title += f', nfit={this_df["n_fit_parameters"].values[0]}'
    plt.title(f'{title}')
    name = str(save_label)
    name += f'mw-{this_df["mw"].values[0]}_s-{this_df["config_sigma"].values[0]}_'
    name += f'{this_df["config_halo_model"].values[0]}_n-{this_df["n_fit_parameters"].values[0]}'
    name += f'_{i}'
    print(name)
    combine_normalized([i], **{"alpha" : 0.3})
    info =""
#     for it,val in this_df.items():
#         if 'prior' in it:
#             print(it,val)
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'save_intermediate', 'earth_shielding', 
                    'nlive']:
        try:
            info += f"\n{str_inf} = %s" % results[i]['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (results[i]['config'][str_inf] / 3600.)
        except KeyError:
            # We were trying to load something that wasn't saved in the config file, ignore it for now.
            pass
    info += '\n\n--prior--'
    for it, val in results[i]['config']['prior'].items():
        if it == 'k':
            continue
        if val['prior_type'] == 'gauss':
            info += f'\n{it} = {val["mean"]} +/- {val["std"]}'
            info += f' | range = {val["range"]}'
        else:
            info += f'\n{it}_range = {val["range"]}'
    info += '\n\n--fit--'
    for it, val in results[i]['res_dict'].items():
        info += f'\n{it} = {val}'
    if '\n' == info[:1]:
        info = info[1:]
        
    ax = plt.gca()
    plt.text(1.6,1, info, transform=ax.transAxes, fontsize=12, bbox=dict(facecolor="white",
                           boxstyle="round"), verticalalignment = 'top')
#     print(info)
    if bin_range:
        plt.xlim(*bin_range[0])
        plt.ylim(*bin_range[1])
    if True:
        plt.savefig(f"{save_dir}/{name}.png", dpi =300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/{name}.pdf", dpi =300, bbox_inches="tight")
#         multinest_corner(results[i], save_dir + name)
    dddm.multinest_corner(results[i], save_dir + name)
    plt.show()
    #     except:
    #         pass