class preprocess:
    
    def __init__(self, dir, file_path):
        os.chdir(dir)
        self.df = pd.read_csv(os.path.join(os.getcwd(), file_path))

    
    def label_df(self):
        col_names = [] #initialize empty list
        
        for index, value in enumerate(self.df.columns): #loop through number of columns in df
            col_name = "slope_" + str(index+1) #concat string with index
            col_names.append(col_name) #add to list
        col_names.pop() #remove last element
        col_names.append("state")

        self.df.columns = col_names #set column names to slope_1, slope_2, ..., slope_29, state
        self.col_names = col_names

        return self.df


    def fishers(self, top_n=5): #filter features by Fisher's criterion
        col_names = self.df.columns 
        df_long = pd.melt(self.df, id_vars=['state'], value_vars=col_names)
        df_mean = df_long.groupby(['state', 'variable'], as_index=False).mean() #mean for each window, grouped by state (0 or 1)
        df_stdev = df_long.groupby(['state', 'variable'], as_index=False).std() #stdev for each window, grouped by state (0 or 1)

        df_summary = pd.merge(df_mean, df_stdev, how='inner', on=['state', 'variable']) #join mean and stdev
        df_summary.columns = ['state', 'slope', 'mean', 'stdev'] #rename columns
        df_summary = pd.pivot(df_summary, columns='state', index='slope', values=['mean', 'stdev']) #pivot wider on state
        df_summary.columns = ['mean_0', 'mean_1', 'stdev_0', 'stdev_1'] #0 for controls, 1 for cases
        df_summary['fishers'] = (df_summary['mean_1'] - df_summary['mean_0'])**2 / (df_summary['stdev_1']**2 + df_summary['stdev_0']**2) #calculate Fisher's criterion for each window
        df_summary = df_summary.sort_values(by='fishers', ascending=False)
        df_summary.reset_index(inplace=True)

        sel_windows = df_summary.head(n=top_n)['slope'].to_list() #list of n windows with highest Fisher's criterion values, default = 5
        sel_windows.append('state') #add 'state' to list
        self.df_fishers = self.df[sel_windows] #subset original dataframe with n windows with highest Fisher's criterion values + state

        return self.df_fishers
