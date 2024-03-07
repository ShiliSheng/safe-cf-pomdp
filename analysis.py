import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator
from scipy.stats import ttest_ind
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats

import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

def get_all_summary(path):
    results = []
    for experiment_setting in os.listdir(path):
        file = path + experiment_setting + "/summary.csv"
        if not os.path.exists(file): continue
        result = pd.read_csv(file, index_col=0)
        shield_level = result["Shield Level"].iloc[0]
        if  shield_level== 0:
            result['Method'] = 'No Shield'
        elif shield_level == 1:
            if result['Failure Rate'].iloc[0] == 0.1:
                continue
            result['Method'] = 'ACP'
        elif shield_level == 2:
            result['Method'] = 'Reactive'
        else:
            continue
            # result['Method'] = 'Max Speed'
        results.append(result)
    r = pd.concat(results, ignore_index = True)
    r.to_csv(path + "all_summary.csv", index=False)
    return r

def format_number(x):
    if x == int(x) or x == round(x, 2):
        return str(x)
    return '{:.{}f}'.format(x, 2).rstrip('0').rstrip('.')
    
def get_stat(path):
    r = get_all_summary(path)
    summary_of_summary = []
    
    for _, df in r.groupby(["Method", "Number of Dynamic Agents"]):
        data ={}
        data["N"] = df["Number of Dynamic Agents"].iloc[0]
        data["Method"] = df["Method"].iloc[0]
        data["P safe"] = df["Percentage of time being safe to dynamic agents"].mean()
        # data['No. Collision']=  df["Number of times being unsafe to static obstacles"].mean()

        
        # data['file'] = path        
        # data["Time"] = df['Action Time spent per action in seconds'].mean().round(3)
        safe_trip = df['Number of times being unsafe to static obstacles'] == 0
        reached_trip = df['Reached Target'] == 1
        safe_reach = df.loc[safe_trip & reached_trip, :]
        data["N_safe_reach"] = len(safe_reach) / len(df)
        data["Reward"] =  df['Cumulative Undiscounted Reward'].mean().round(1)

        summary_of_summary.append(pd.DataFrame([data], columns = data.keys()))
    summary = pd.concat(summary_of_summary, ignore_index = True)
    # summary["alpha"] = summary["alpha"].apply(format_number)
    summary["P safe"] = summary["P safe"].apply(format_number)
    # summary = summary.sort_values(by = [ "N"])
    # print(summary.values)
    
    with open(path + 'stat.txt', 'w') as f:
        for n in sorted(summary.N.unique()):
            for method in ['No Shield', "Reactive", "ACP", ]:
                rows = summary.loc[summary.N==n, :]
                rows = rows.loc[rows.Method==method, :]
                for row in rows.values:
                    f.write(";".join(str(item) for item in row) + "\n")

    # latex_table = summary.to_latex(index = False)
    # print(latex_table)
    # with open('merged_cells.txt', 'w') as f:
    #     f.write(latex_table)
    
    # hue_order = ["No Shield", "Reactive", "Max Speed", "ACP (0.05)", "ACP (0.1)"]
    hue_order = ["No Shield", "Reactive", "ACP", ]
    x = 'Number of Dynamic Agents'
    hue = 'Method'

    y = 'Minimum Distance to Agents'
    boxplot(path, r, x = x, y = y, hue = hue,
            xlabel = 'Number of Pedestrians' , ylabel = 'Minimum Distance (m)', 
            fig_name = '{}-Distance.jpg'.format(r.loc[0,"Model"]), ncol = 4, hline = 2, hue_order=hue_order)
    
    y = 'Cumulative Undiscounted Reward'
    boxplot(path, r, x = x, y = y, hue = hue,
            xlabel = 'Number of Pedestrians' , ylabel = 'Undiscounted Reward', 
            fig_name = '{}-Undiscounted Reward.jpg'.format(r.loc[0,"Model"]), ncol = 3, hue_order=hue_order)
    
    y = 'Cumulative Discounted Reward'
    boxplot(path, r, x = x, y = y, hue = hue,
            xlabel = 'Number of Pedestrians' , ylabel = 'Discounted Reward', 
            fig_name = '{}-Discounted Reward.jpg'.format(r.loc[0,"Model"]), ncol = 3, hue_order = hue_order)
    return summary

def t_test(data, x, y, hue):
    p_values = {}
    for x_item in data[x].unique():
        for hue_i, hue_item_i in enumerate(data[hue].unique()):
            for hue_j, hue_item_j in enumerate(data[hue].unique()):
                if hue_j > hue_i:
                    subset1 = data[(data[x] == x_item) & (data[hue] == hue_item_i)][y]
                    subset2 = data[(data[x] == x_item) & (data[hue] == hue_item_j)][y]
                    _, p_value = ttest_ind(subset1, subset2)
                    p_values[(x_item, hue_item_i, hue_item_j)] = p_value
                    if (p_value < 0.05):
                        print(x, y, hue, (x_item, hue_item_i, hue_item_j), p_value, "___++")
    return p_values

def perform_one_way_anova(df, x, y, hue, path):
    df = df.rename(columns={x: x.replace(" ", "_")})
    df = df.rename(columns={y: y.replace(" ", "_")})
    df = df.rename(columns={hue: hue.replace(" ", "_")})
    x, y, hue = x.replace(" ", "_"), y.replace(" ", "_"), hue.replace(" ", "_")

    formula = f"{y} ~ C({x}) + C({hue})"
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    tukey_result = pairwise_tukeyhsd(df[y], df[hue])
    # print( anova_table['F'][0], anova_table['PR(>F)'][0], )
    # print(anova_table, tukey_result)
    with open(path + 'stat.txt', 'a') as f:
        f.write(str(anova_table))   
        f.write(str(tukey_result)) 
    return anova_table, tukey_result

def boxplot(path, dataframe, x, y, hue, xlabel, ylabel, fig_name, ncol = 1, hline = -float("inf"), fig_format = 'jpg', hue_order = []):
    # p_values = t_test(dataframe, x, y, hue)
    perform_one_way_anova(dataframe, x, y, hue, path)

    custom_palette = {"No Shield": (206/255, 170/255, 210/255),
                   "Reactive": (110/255, 156/255, 114/255), 
                   "ACP": (213/255, 159/255, 63/255), 
                #    "ACP (0.1)": (95/255, 116/255, 160/255),
                #    "Max Speed": (195/255, 116/255, 160/255),
                  }
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 13
    plt.figure(figsize=(5, 4))
    box_plot = sns.boxplot(data = dataframe, x = x, y = y, hue = hue, 
                           hue_order = hue_order, 
                           width = 0.5, 
                           palette=custom_palette, linewidth = 0.6,
                           fliersize=0.5
                           )
    if hline != -float("inf"):
        plt.axhline(y=hline, color='r', linestyle='dotted', label= 'Safe Distance')
    
    # box_pairs =[]
    # hue_unique = dataframe[hue].unique()
    # for x_item in dataframe[x].unique():
    #     for hue_i in range(len(hue_unique)):
    #         for hue_j in range(hue_i + 1, len(hue_unique)):
    #             box_pairs.append(((x_item, hue_unique[hue_i]), (x_item, hue_unique[hue_j])))
    # add_stat_annotation(box_plot, data=dataframe, x=x, y=y, hue=hue, 
    #                     box_pairs=box_pairs,
    #                 test='t-test_ind', loc='inside', verbose=2)

    legend = box_plot.legend(
                            loc='upper center', 
                             bbox_to_anchor=(0.5, 1.15 ), 
                             ncol = ncol,
                             columnspacing = 1.2,
                             handletextpad=0.4,
                            handlelength = 0.7)
    legend.get_frame().set_facecolor('none')  # Set legend background color
    legend.get_frame().set_linewidth(0)        # Remove legend border
    
    # box_plot.set_title('Box plot of distance grouped by categorical columns and N_agents')
    box_plot.set_xlabel(xlabel)
    box_plot.set_ylabel(ylabel)
    
    plt.tight_layout()
    plt.savefig(path + fig_name,  dpi=300 , bbox_inches="tight", format=fig_format)
    # plt.show()
    plt.close()

def get_statistics(path):
    np.set_printoptions(precision=2, suppress=True)
    result = []
    record_min_distance = defaultdict(list)
    record_reward = defaultdict(list)
    N_agents = 100
    for experiment_setting in os.listdir(path):
        # if not os.path.isdir(path + experiment_setting ): continue
        if not os.path.exists(path + experiment_setting + '/summary.csv'): continue
        print(experiment_setting)
        file = path + experiment_setting + "/summary.csv"
        day_time = ''.join(experiment_setting.split("-")[8:])
        df = pd.read_csv(file, index_col=0)
        n_episodes = len(df)
        # for i_episode in range(n_episodes):
        #     episode_file = os.path.join(path, experiment_setting, "Episode-{}.pkl".format(i_episode))
        #     with open(episode_file, 'rb') as episode_file_pkl:
        #         episode_data = pickle.load(episode_file_pkl)
        #     index = 0
        #     cnt_action_taken = 0
        #     cumulative_time = 0
        #     max_plan_time = 0
        #     cnt_close_to_agents = 0
        #     while index+1 < len(episode_data):
        #         if episode_data[index+1]["Action Step"] > 1:
        #             plan_time = episode_data[index+1]["Clock Time"] - episode_data[index]["Clock Time"]
        #             cumulative_time += plan_time
        #             cnt_action_taken += 1
        #             max_plan_time = max(max_plan_time, plan_time)
        #             if episode_data[index + 1]["Current Minimum Distance to Agents"] < episode_data[index + 1]["Safe Distance"]:
        #                 cnt_close_to_agents += 1
        #         index+=1
        #     df.loc[i_episode, "max_plan_time"] = max_plan_time
        #     df.loc[i_episode, "average_plan_time"] = cumulative_time / cnt_action_taken
        #     df.loc[i_episode, "Number of being unsafe distance to agents"] = cnt_close_to_agents

        # for i in range(len(df)):
        #     df.loc[i, 'Trip safe with pedetrains'] = 1 if df.loc[i,"Minimum Distance to Agents"] >= df.loc[i,"Safe Distance"] else 0
        #     df.loc[i,'Trip reached and safe with pedestrians'] = 1 if df.loc[i, 'Reached Target'] and df.loc[i, 'Trip safe with pedetrains'] else 0
        #     df.loc[i, 'Trip safe with obstacles'] = 1 if df.loc[i,"Number of Unsafe State"]  == 0 else 0
        #     df.loc[i,'Trip reached and safe with obstacles'] = 1 if df.loc[i, 'Reached Target'] and df.loc[i, 'Trip safe with obstacles'] else 0
        #     df.loc[i, "Trip reached and safe with both"] = 1 if df.loc[i,'Trip reached and safe with pedestrians'] and df.loc[i,'Trip reached and safe with obstacles'] else 0
        
        data = {
            "Shield Level":                                 df["Shield Level"].iloc[0],
            "Predict Horizon":                              df["Predict Horizon"].iloc[0] if df["Shield Level"].iloc[0] ==1 else "-",
            "Failure Rate":                                 df["Failure Rate"].iloc[0] if df["Shield Level"].iloc[0] ==1 else "-",
            "Number of Dynamic Agents":                     df["Number of Dynamic Agents"].iloc[0],
            "Percentage of time being safe to dynamic agents":          df["Percentage of time being safe to dynamic agents"].mean(),
            'Rate of Reaching Target':                                  df['Reached Target'].mean(),
            "Mean of Minimum Distance to Agents":            df["Minimum Distance to Agents"].mean(),
            "SD of Minimum Distance to Agents":            df["Minimum Distance to Agents"].std(),
            "Min of Minimum Distance to Agents":            df["Minimum Distance to Agents"].min(),
            'Cumulative Undiscounted Reward':               df['Cumulative Undiscounted Reward'].mean(),
            "Number of Unsafe Action":                      df["Number of Unsafe Action"].mean(),
            "Number of Action Steps":                       df["Number of Action Steps"].mean(),
            'Average Number of times being unsafe to static obstacles': df["Number of times being unsafe to static obstacles"].mean(),
            'Average Number of times being unsafe to dynamic agents':   df["Number of times being unsafe to dynamic agents"].mean(),
            "Percentage of time being safe to static agents":           df["Percentage of time being safe to static agents"].mean(),
            'Action Time spent per action in seconds':                  df['Action Time spent per action in seconds'].mean(),
            "Setting":                                      experiment_setting,
            "Episodes":                                     len(df),
            # 'Average number of Collision with Static Obstalces':    df['Number of Unsafe State'].mean(),
            # 'Average number of Unsafe Distance to agents':          df["Number of being unsafe distance to agents"].mean(),
            # 'Rate of Reach and safe with pedestrains':              df['Trip reached and safe with pedestrians'].mean(),
            # 'Rate of Reach and safe with obstacles':                df['Trip reached and safe with obstacles'].mean(),
            # 'Rate of Reach and safe with both':                     df["Trip reached and safe with both"].mean(),
            # 'Rate safe with pedestrains given reached':             (df['Trip reached and safe with pedestrians'].sum()) / (df['Reached Target'].sum()),
            # 'Rate safe with obstacles given reached':               (df['Trip reached and safe with obstacles'].sum()) / (df['Reached Target'].sum()),
            # 'Rate safe with both given reached':                    (df['Trip reached and safe with both'].sum()) / (df['Reached Target'].sum()),
            # 'Average of average plan time':     df['average_plan_time'].mean(),
            # 'Max of max plan time':             df['max_plan_time'].max(),
        }
        if data["Shield Level"] != 0:
            setting = "S{}-H{}-a{}".format(data["Shield Level"], data["Predict Horizon"], data["Failure Rate"])
        else:
            setting = 'S0'

        min_distance = pd.DataFrame()
        min_distance[setting] = df["Minimum Distance to Agents"]
        record_min_distance[data["Number of Dynamic Agents"]].append(min_distance)

        reward = pd.DataFrame()
        reward[setting] = df["Cumulative Undiscounted Reward"]
        record_reward[data["Number of Dynamic Agents"]].append(reward)

        result.append(pd.DataFrame([data], columns = data.keys()))
        print(df["Percentage of time being safe to static agents"].mean(), data["Percentage of time being safe to dynamic agents"])

    res = pd.concat(result, ignore_index = True)
    res = res.sort_values(by = ["Shield Level", "Failure Rate", "Number of Dynamic Agents"])
    # print(res.values, res.columns)
    res.to_csv( path + "stat2.csv", index = False)
    
    for N_agents in record_min_distance:
        for record, name in [(record_min_distance[N_agents], "Minimum Distance to Pedestrians"), (record_reward[N_agents], "Reward")]:
            for column in record.columns:
                box_plot = sns.boxplot(x=N_agents, y=record, data=record)

            df = pd.concat(record, axis= 1)
            # df = df.reindex(sorted(df.columns), axis=1)
            df.boxplot()
            plt.grid(False)
            plt.xticks(rotation=30)
            plt.ylabel(name)
            plt.title("Number of pedestrians {}".format(N_agents))
            plt.axhline(y=0.5, color='r', linestyle='--', label='Safety Distance')
            plt.savefig(path + name + "_N{}.png".format(N_agents), dpi = 300, bbox_inches= "tight")
            plt.show()

            f_statistic, p_value = f_oneway(*[df[col] for col in df.columns])
            print(f'F-statistic: {f_statistic}\nP-value: {p_value}')

            # Check the p-value to determine if there is a significant difference
            alpha = 0.05
            if p_value < alpha:
                print("Reject the null hypothesis: There is a significant difference between groups.")
            else:
                print("Fail to reject the null hypothesis: There is no significant difference between groups.")
            # Tukey's HSD post hoc test
            melted_df = pd.melt(df)
            posthoc = pairwise_tukeyhsd(melted_df['value'], melted_df['variable'], alpha=0.05)

            # Print the results of the post hoc test
            print(posthoc)

if __name__ == "__main__":
    t = "_0126_good"
    # t = ""
    ETH_path = './results{}/Obstacle-ETH-0-0-22-22/'.format(t)
    death = "./results{}/Obstacle-SDD-deathCircle-video1-0-0-50-60/".format(t)
    bookstore ="./results{}/Obstacle-SDD-bookstore-video1-0-0-50-45/".format(t)
    s1 = get_stat(ETH_path)
    s2 = get_stat(death)
    s3 = get_stat(bookstore)
    s = pd.concat([s1, s2, s3])
    # print(s)
    # for _, df in s.groupby("Method"):
    #     print(df.Method.unique(), df.Time.mean())
    # ETH_path = './results/Obstacle-ETH-0-0-22-22/'
    # get_stat(ETH_path)
    # get_statistics(path = )
    # get_statistics(path = )
    # df = pd.read_table(path + "stat.csv", sep = ",")
    # print(df.columns)
    # # # Assuming df is your DataFrame
    # # df = df.loc[df["Shield Level"] == 1, :]
    # X = df[["Shield Level", 'Failure Rate', 'Predict Horizon', 'Number of Dynamic Agents']]
    
    # cols = [
    #    'Average Minimum Distance to Agents',
    #    'Min of Minimum Distance to Agents', 'Cumulative Undiscounted Reward',
    #    'Number of Unsafe Action',
    #    'Average number of Collision with Static Obstalces',
    #    'Average number of Unsafe Distance to agents',
    #    'Rate of Reaching Target', 'Rate of Reach and safe with pedestrains',
    #    'Rate of Reach and safe with obstacles',
    #    'Rate of Reach and safe with both',
    #    'Rate safe with pedestrains given reached',
    #    'Rate safe with obstacles given reached',
    #    'Rate safe with both given reached',
    #    'Action Time spent per action in seconds',
    #    'Average of average plan time', 'Max of max plan time']
    # for col in cols:
    #     print(col,"---")
    #     y = df[col]
        
    #     # Add a constant term to the independent variables
    #     X = sm.add_constant(X)

    #     # Fit the logistic regression model
    #     try:
    #         model = sm.Logit(y, X)
    #         result = model.fit()
    #         # Display the summary of the regression
    #         print("+++++++++++++++++++++++++++++++++++++++")
    #         print(col)
    #         print(result.summary())
    #     except:
    #         continue
    # pass