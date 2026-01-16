# -*- coding: utf-8 -*-
"""
STREAM (Aerts, 1999)
Modified and Converted to MATLAB by Hylke Beck
Modified by Hans de Moel
Converted to Python by Timothy Tiggeloven en Joris Westenend
Modified by Timothy Tiggeloven
"""
# Modules and settings
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mplcursors
import numpy as np
import os
import pandas as pd
from requests import patch
from tqdm import tqdm

import panel.widgets as pnw
import panel as pn
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

from pysheds.grid import Grid
from pysheds.sview import Raster

# Hide warnings regarding nans in conditional statements
np.seterr(invalid='ignore')

class Setup_model():
    def __init__(self, dischargepointrow, dischargepointcolumn, pixelsize, timestep, C, Cropf,
                 Wholdn, Aw, Gw, Snow, Precipitation, Temperature, Heat, A, grid, fdir, dirmap,
                 month_list, year_list, timestart=0, timeend=100, HEATcal=1, WHOLDNcal=1,
                 MELTcal=10, CROPFcal=1, TOGWcal=0.4, Ccal=1, beginmonth=0):
        # General
        self.dischargepointrow = dischargepointrow
        self.dischargepointcolumn = dischargepointcolumn
        self.pixelsize = pixelsize
        self.timestep = timestep
        self.dirmap = dirmap
        self.month_list = month_list
        self.year_list = year_list

        # Input maps
        self.C = C
        self.Cropf = Cropf
        self.Wholdn = Wholdn
        self.Aw = Aw
        self.Gw = Gw
        self.Snow = Snow
        self.Precipitation = Precipitation
        self.Temperature = Temperature
        self.Heat = Heat
        self.A = A
        self.grid = grid
        self.fdir = fdir

        # Calibration initialization
        self.HEATcal = HEATcal     # Not used in this practical: default=1 [>0]
        self.WHOLDNcal = WHOLDNcal # Water holding capacity of the soil: default=1 [>0]
        self.MELTcal = MELTcal     # How fast snow melts: default=10 [>0]
        self.CROPFcal = CROPFcal   # Parameter steering the evapotranspiration: default=1 [>0]
        self.TOGWcal = TOGWcal     # fraction going to gw and direct runoff: default=0.4 [0-1]
        self.Ccal = Ccal           # Parameter steering how fast groundwater flows: default=1 [>1]
    
    def model_flow(self, timestart=0, timeend=100, P_change=1.0, T_change=0.0):
        # Create dataframes and lists to store results in
        self.TimeseriesQsec = pd.DataFrame(index=range(timestart, timeend), columns=['Qsec'])
        self.TimeseriesPre = pd.DataFrame(index=range(timestart, timeend), columns=['Pre'])
        self.TimeseriesPre_act = pd.DataFrame(index=range(timestart, timeend), columns=['Pre'])
        self.TimeseriesGw = pd.DataFrame(index=range(timestart, timeend), columns=['Gw'])
        self.TimeseriesTmp = pd.DataFrame(index=range(timestart, timeend), columns=['Tmp'])
        self.TimeseriesPe = pd.DataFrame(index=range(timestart, timeend), columns=['Pe'])
        self.TimeseriesExcess = pd.DataFrame(index=range(timestart, timeend), columns=['Excess'])
        self.TimeseriesAw = pd.DataFrame(index=range(timestart, timeend), columns=['Aw'])
        self.TimeseriesRunoff = pd.DataFrame(index=range(timestart, timeend), columns=['Runoff'])
        self.Qsec_list = []
        self.Snow_list = []
        self.Gw_list = []
        self.Aw_list = []
        self.Pe_list = []
        self.Melt_list = []
        
        # retrieve initial maps of basin water storage
        Snow = self.Snow.copy()
        Aw = self.Aw.copy()
        Gw = self.Gw.copy()

        self.timestart = timestart
        self.timeend = timeend

        # Start iteration
        for i in tqdm(range(timestart, timeend)):
            # Reading climate scenario data
            Prei = self.Precipitation[i, :, :].clip(min=0)  # No negative precipitation           
            Prei[np.isnan(Prei)] = 0                        # No data values set to 0
            #ADDED PRECIPITATION CHANGE #######################################################
            Prei = Prei * P_change
            ###################################################################################
            Pre = np.kron(Prei, np.ones((30, 30)))

            Tmpi = self.Temperature[i, :, :]
            #ADDED TEMPERATURE CHANGE #########################################################
            Tmpi = Tmpi + T_change
            ###################################################################################
            Tmp = np.kron(Tmpi, np.ones((30, 30)))

            self.TimeseriesPre_act['Pre'].iloc[i - timestart] = np.nanmean(Pre)

            # Calculate snow fall, snow cover storage and snow melt
            Snowfall = np.copy(Pre)
            Snowfall[Tmp > 0] = 0  # At high temperatures no snow falls
            Snow = Snow + Snowfall
            Melt = self.MELTcal * Tmp  # Snow will melt at high temperatures
            Melt[Tmp < 0] = 0
            Melt = np.minimum(Snow, Melt)
            Snow -= Melt
            Pre = Pre - Snowfall + Melt

            # Calculate potential evapotranspiration using Thornthwaite
            Pe = np.full_like(Tmp, np.nan)
            Heat = self.Heat * self.HEATcal
            Pe[Tmp<26.5] = 16 * ((10 * (Tmp[Tmp < 26.5] / Heat[Tmp < 26.5])) ** self.A[Tmp < 26.5])
            Pe[Tmp>=26.5] = -415.85 + (32.24 * Tmp[Tmp >= 26.5]) - (0.43 * (Tmp[Tmp >= 26.5] ** 2))    
            Pe[Tmp<=0] = 0
            Pe = Pe * self.Cropf * self.CROPFcal

            # Calculate soil storage according to Thornthwaite-Mather
            Wholdn = self.Wholdn * self.WHOLDNcal
            Peff = Pre - Pe
            Aw_1 = np.copy(Aw)  # available water in last interation
            Excess = np.zeros_like(Peff)
            #Soil is wetting below capacity
            below_cap = (np.add(Aw_1, Peff)) <= Wholdn
            Aw[below_cap] = Aw_1[below_cap] + Peff[below_cap]
            Excess[below_cap] = 0
            #Soil is wetting above capacity
            above_cap = (np.add(Aw_1,Peff))> Wholdn
            Excess[above_cap] = Aw_1[above_cap] + Peff[above_cap] - Wholdn[above_cap]
            Aw[above_cap] = Wholdn[above_cap]
            #Soil is drying    
            Aw[Peff <= 0] = Aw_1[Peff <= 0] * np.exp(Peff[Peff <= 0] / Wholdn[Peff <= 0])
            Excess[Peff <= 0] = 0

            # Separate direct from delayed runoff (seepage to groundwater)
            Runoff = self.TOGWcal * Excess
            Togw = Excess - Runoff

            # Calculate volume of groundwater and baseflow
            Gw = Gw + Togw
            Sloflo = Gw / (self.C * self.Ccal)
            Gw -= Sloflo

            # Calculate discharge (snow melt + runoff + baseflow),
            # and create map of total monthly discharge per cel
            Dschrg = Runoff + Sloflo

            # Recalculate mm -> m^3; accumulate discharge and calculate m^3/sec
            Dschrg = (Dschrg / 1000) * self.pixelsize * self.pixelsize  # conversion from mm to m^3
            weights = Raster(Dschrg, viewfinder=self.fdir.viewfinder, metadata={'nodata': np.nan})

            Q = self.grid.accumulation(self.fdir, dirmap=self.dirmap, weights=weights)
            Qsec = Q / self.timestep  # conversion from m^3 to m^3/sec
            Qstation = Qsec[self.dischargepointrow, self.dischargepointcolumn]

            # Store results in list
            self.TimeseriesPre['Pre'].iloc[i - timestart] = np.nanmean(Pre)
            self.TimeseriesGw['Gw'].iloc[i - timestart] = np.nanmean(Gw)
            self.TimeseriesTmp['Tmp'].iloc[i - timestart] = np.nanmean(Tmp)
            self.TimeseriesPe['Pe'].iloc[i - timestart] = np.nanmean(Pe)
            self.TimeseriesExcess['Excess'].iloc[i - timestart] = np.nanmean(Excess) 
            self.TimeseriesAw['Aw'].iloc[i - timestart] = np.nanmean(Aw)
            self.TimeseriesRunoff['Runoff'].iloc[i - timestart] = np.nanmean(Runoff)
            self.TimeseriesQsec['Qsec'].iloc[i - timestart] = np.nanmean(Qstation)
            self.Qsec_list.append(Qsec)
            self.Snow_list.append(Snow.copy())
            self.Gw_list.append(Gw.copy())
            self.Aw_list.append(Aw.copy())
            self.Pe_list.append(Pe.copy())
            self.Melt_list.append(Melt.copy())
        self.df = pd.concat([self.TimeseriesQsec['Qsec'], self.TimeseriesRunoff['Runoff'],
                             self.TimeseriesAw['Aw'], self.TimeseriesExcess['Excess'],
                             self.TimeseriesPe['Pe'], self.TimeseriesPre['Pre'],
                             self.TimeseriesTmp['Tmp'], self.TimeseriesGw['Gw']], axis=1)

    def locate_station(self, timestep=0):
        fig, ax = plt.subplots()
        im_ax = ax.imshow(self.Qsec_list[timestep])
        mplcursors.cursor(hover=True)
        plt.show()

    def get_single_annual_avg(self):
        sim_time = range(self.timestart, self.timeend)
        start_year = self.year_list[0]
        year = [start_year + x // 12 for x in sim_time]
        year_df = pd.DataFrame(year, columns=['year'])

        merged_data = pd.concat([self.Precipitation, year_df], axis=1)

        annual_avg = merged_data.groupby("year").mean().reset_index()
        single_annual_avg = annual_avg.mean(numeric_only=True)

        return single_annual_avg
    
    ############# Visualization functions ###################################################################################
    def animate(self, rasters=False, timestart=False, timeend=False):
        def animate_func(i):
            if i % 10 == 0:
                print('.', end ='')
            im.set_array(rasters[i])
            if i > 0:
                ax.texts[i - 1].set_visible(False)
            texts[i].set_text(f'Year: {self.year_list[timestart + i]}, Month: {self.month_list[timestart + i]}')
            return [im]

        if not rasters:
            rasters = self.Qsec_list
        if not timestart and not timeend:
            timestart = self.timestart
            timeend = self.timeend

        # First set up the figure, the axis, and the plot element we want to animate
        fig, ax = plt.subplots(figsize=(6, 5))

        im = plt.imshow(rasters[0], aspect='auto', animated=True, cmap='hot')
        texts = [ax.text(0.1, 1.02, '', fontsize=20, transform=ax.transAxes) for i in range(len(rasters))]

        ani = animation.FuncAnimation(fig, animate_func, interval=200, blit=True,
                                      repeat_delay=1000, frames=len(rasters))

        plt.close(ani._fig)
        return HTML(ani.to_html5_video())    
    
    def plot_series(self):
        def mpl_plot(variable):
            fig = Figure()
            FigureCanvas(fig) # not needed in mpl >= 3.1
            ax = fig.add_subplot()
            self.df[variable].plot(ax=ax)
            fig.patch.set_facecolor('white')
            return fig
        
        pn.extension()
        variable  = pnw.RadioButtonGroup(name='variable', value='Aw', options=list(self.df.columns))
        # window  = pnw.IntSlider(name='window', value=10, start=1, end=60)

        @pn.depends(variable)
        def reactive_outliers(variable):
            return mpl_plot(variable)

        widgets   = pn.Column("<br>\n# Area", variable)
        # widgets   = pn.Column("<br>\n# Room occupancy", variable)
        occupancy = pn.Row(reactive_outliers, widgets)
        return occupancy
    
    def plot_hydrograph(self):
        # months = [(x % 12) + 1 for x in range(self.timestart, self.timeend)]
        # Q_simulated = self.TimeseriesQsec.copy()
        # Q_simulated["months"] = months
        months = [x+1 for x in self.TimeseriesQsec.index]

        fig, ax = plt.subplots()
        ax.plot(months, self.TimeseriesQsec['Qsec'], color="#569AD8")
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Discharge (m³/s)')
        ax.set_title('Hydrograph')
        ax.grid(visible=True, which="major", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)
        
        return fig

    def plot_average_discharge(self):
        sim_time = range(self.timestart, self.timeend)
        months = [(x % 12) + 1 for x in sim_time]
        Q_simulated = self.TimeseriesQsec.copy()
        Q_simulated["months"] = months
        avg_monthly_discharge = Q_simulated.groupby("months").mean()

        fig, ax = plt.subplots()
        
        ax.bar(avg_monthly_discharge.index, avg_monthly_discharge['Qsec'], color="#569AD8")
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Discharge (m³/s)')
        ax.set_title('Average Monthly Discharge')
        ax.grid(visible=True, which="major", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

        return fig
    
    def plot_cumulative_discharge(self):
        volume = (self.TimeseriesQsec * self.timestep).cumsum()

        fig, ax = plt.subplots()
        ax.plot(volume.index, volume['Qsec'], color="#569AD8")
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Cumulative Discharge (m³)')
        ax.set_title('Cumulative Discharge Over Time')
        ax.grid(visible=True, which="major", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

        return fig
    
    def plot_annual_maxima(self):
        sim_time = range(self.timestart, self.timeend)
        start_year = self.year_list[0]
        year = [start_year + x // 12 for x in sim_time]
        year_df = pd.DataFrame(year, columns=['year'])
        merged_data = pd.concat([self.TimeseriesQsec['Qsec'], year_df], axis=1)

        annual_maxima = merged_data.groupby("year").max().reset_index()

        fig, ax = plt.subplots()
        ax.bar(annual_maxima['year'], annual_maxima['Qsec'], color="#569AD8")
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Maximum Discharge (m³/s)')
        ax.set_title('Annual Maximum Discharge')
        ax.grid(visible=True, which="major", axis="y", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

        return fig
    
    def plot_frequency_curve(self):
        sim_time = range(self.timestart, self.timeend)
        months = [(x % 12) + 1 for x in sim_time]
        Q_simulated = self.TimeseriesQsec.copy()
        Q_simulated["months"] = months

        sorted_flows = np.sort(self.TimeseriesQsec['Qsec'])[::-1]
        probabilities = np.arange(1, len(sorted_flows) + 1) / (len(sorted_flows) + 1)

        fig, ax = plt.subplots()
        ax.plot(sorted_flows, probabilities, color="#569AD8")
        ax.set_xlabel('Discharge (m³/s)')
        ax.set_ylabel('Probability')
        ax.set_title('Frequency Curve')
        ax.grid(visible=True, which="major", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

        return fig

    def plot_flow_variability(self):
        sim_time = range(self.timestart, self.timeend)
        months = [(x % 12) + 1 for x in sim_time]
        Q_simulated = self.TimeseriesQsec.copy()
        Q_simulated["months"] = months

        labels = ["All", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots()

        data_per_month = [Q_simulated.loc[Q_simulated["months"] == m, "Qsec"].values
                        for m in range(1, 13)]
        bplot = ax.boxplot(data_per_month, positions=range(1, 13),
                        patch_artist=True, widths=0.6)

        for box in bplot['boxes']:
            box.set_facecolor('#569AD8')

        extra = ax.boxplot([Q_simulated["Qsec"].values], positions=[0], patch_artist=True, widths=0.6, boxprops=dict(facecolor='#215B8F'))
        
        ax.set_xticks(range(0, 13))
        ax.set_ylabel('Discharge (m³/s)')
        ax.set_xticklabels(labels)
        ax.set_title('Flow Variability')
        ax.grid(visible=True, which="major", axis="y", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

        return fig
    
    def plot_flow_duration_curve(self):
        sorted_flows = np.sort(self.TimeseriesQsec['Qsec'])[::-1]
        exceedance_probabilities = np.arange(1, len(sorted_flows) + 1) / len(sorted_flows) * 100
        
        fig, ax = plt.subplots()
        ax.plot(exceedance_probabilities, sorted_flows, color="#569AD8")
        ax.set_xlabel('Exceedance Probability (%)')
        ax.set_ylabel('Discharge (m³/s)')
        ax.set_title('Flow Duration Curve')
        ax.set_xlim(0, 100)
        ax.grid(visible=True, which="major", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)
        
        return fig
    
############## Additional functions ###################################################################################

def set_dates(start_year, end_year, start_month, end_month):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
             'September', 'October', 'November', 'December']
    month_list = months[start_month:]
    year_list = [start_year] * len(month_list)
    month_list.extend(months * (end_year - start_year - 1))
    for i in range(end_year - start_year - 1):
        year_list.extend([i + start_year + 1] * 12)
    month_list.extend(months[:end_month])
    year_list.extend([start_year] * len(months[:end_month]))
    return month_list, year_list

def yearly_average_comparison(start_year, timestart, timeend, list_of_runs, list_of_labels):
    sim_time = range(timestart, timeend)
    year = [start_year + x // 12 for x in sim_time]

    merged_data = pd.concat(list_of_runs, axis=1)
    merged_data.columns = list_of_labels
    merged_data["year"] = year

    yearly_sum = merged_data.groupby("year").sum().reset_index()
    yearly_avg = yearly_sum.mean()
    yearly_avg = yearly_avg.drop('year')

    fig, ax = plt.subplots()
    ax.bar(yearly_avg.index, yearly_avg.values, zorder=2, color="#569AD8")
    ax.set_xticks(yearly_avg.index)
    ax.set_xticklabels(list_of_labels, rotation=45, ha="right")
    ax.set_ylabel('Annual Average Discharge (m³/s)')
    ax.set_title('Annual Average Discharge Comparison')

    ax.grid(visible=True, which="major", axis="y", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

    return fig

def monthly_average_comparison(timestart, timeend, list_of_runs, list_of_labels, list_of_colors):

    fig, ax = plt.subplots()

    for i, run in enumerate(list_of_runs):
        if i > 0:
            max_run = avg_monthly_discharge['Qsec'].max()
        else:
            max_run = 0

        sim_time = range(timestart, timeend)
        months = [(x % 12) + 1 for x in sim_time]
        Q_simulated = run.copy()
        Q_simulated["months"] = months

        avg_monthly_discharge = Q_simulated.groupby("months").mean()

        ax.plot(avg_monthly_discharge.index + i*0.1, avg_monthly_discharge['Qsec'], label=list_of_labels[i], color=list_of_colors[i], zorder=2)

        max_run_new = avg_monthly_discharge['Qsec'].max() 
        max_all = max(max_run, max_run_new)
    
    unique_months = sorted(Q_simulated["months"].unique())
    ax.set_xticks(unique_months)
    ax.set_xticklabels(unique_months) 
    ax.set_xlabel('Month')
    ax.set_ylim(0, max_all * 1.1)
    ax.set_ylabel('Average Discharge (m³/s)')
    ax.set_title('Monthly Average Discharge Comparison')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(visible=True, which="major", axis="y", color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.2)

    return fig

def fig_download(fig, filename='figure.png'):
    output_dir = "../results"

    fig.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/{filename}.svg", dpi=300, bbox_inches="tight")

