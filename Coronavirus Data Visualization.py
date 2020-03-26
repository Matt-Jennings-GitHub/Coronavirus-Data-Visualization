# Modules
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.patches as patches
import cartopy.io.shapereader as shpreader
import itertools
import numpy as np
from random import random
import pandas as pd
from iso3166 import countries
from time import sleep
from datetime import date as dtdate
import yfinance as yf
from colorsys import rgb_to_hls, hls_to_rgb

# Stock Data
nat_indicies = {'^TA125.TA':['TA-125',(32.0853, 34.7818)],'^BVSP':['IBOVESPA',(-23.5505, -46.6333)],'^BSESN':['BSE SENSEX',(19.0760, 72.8777)],'^AXJO':['ASX 200',(-33.8688, 151.2093)],'^STI':['STI Index',(1.3521, 103.8198)],'IMOEX.ME':['MOEX Russia Index',(55.7558, 37.6173)],'^FTSE':['FTSE 100',(51.5074, 0.1278)],'^GSPC':['S&P 500',(40.7128, -74.0060)],'000001.SS':['SSE Composite Index',(31.2304, 121.4737)],'^HSI':['Hang Seng Index',(22.3193, 114.1694)],'^N225':['NIKKEI 225',(35.6762, 139.6503)]}#'^GSPTSE':['TSX Composite index',(43.6532, -79.3832)]
ind_dfs = {}
for ind in nat_indicies:
    df_ind = yf.download(ind,'2019-12-31',dtdate.today())
    initial_price = df_ind['Close'].iloc[0]
    dates = pd.date_range('2019-12-31',dtdate.today())
    df_ind.index = pd.DatetimeIndex(df_ind.index)
    df_ind = df_ind.reindex(dates, fill_value=0)
    df_ind.iloc[0, df_ind.columns.get_loc('Close')] = initial_price
    for i in range(1,len(df_ind['Close'])): 
        if df_ind['Close'].iloc[i] == 0:
            df_ind.iloc[i, df_ind.columns.get_loc('Close')] = df_ind.iloc[i-1, df_ind.columns.get_loc('Close')]
            
    df_ind.reset_index(inplace=True)
    df_ind['Date'] = df_ind['index'].dt.strftime('%d/%m/%Y')
    ind_dfs[ind] = df_ind
    

# Corona Data
datafile_covid = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{}.xlsx'.format(dtdate.today())
df = pd.read_excel(datafile_covid, names = ['Date', 'Day', 'Month', 'Year','Cases','Deaths','Country','ID','Pop'], skiprows = 1)

datafile_coords = 'CountryCoords.csv'
df_coords = pd.read_csv(datafile_coords, names = ['Code', 'Lat', 'Long', 'Name'], skiprows=1)

codes = {'UK':'GBR','AL':'ALB','AG':'ATG','AR':'ARG','BS':'BHS','BD':'BGD','BJ':'BEN','BT':'BTN','BO':'BOL','BA':'BIH','BN':'BRN','BG':'BGR','BF':'BFA','CM':'CMR','EL':'GRC','NA':'NAM','PS':'PSE','SO':'SOM'}
for country in countries: # Convert to 3 letter country codes
    codes[country.alpha2] = country.alpha3
codes['UK'] = 'GBR'
for code in codes:
    df['ID'] = df['ID'].replace(code,codes[code])
    df_coords['Code'] = df_coords['Code'].replace(code,codes[code])

df = df[df['Country'] != 'Cases_on_an_international_conveyance_Japan'] # Drop cruiseship rows
df['Year'] = df['Year'].replace(2019,1) # Convert year to number
df['Year'] = df['Year'].replace(2020,2)
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
df = df.sort_values(by=['Country','Year','Month','Day'],ascending=[True,True,True,True])

datetimes = pd.date_range('31-12-2019', dtdate.today()).strftime('%d/%m/%Y').tolist()
dates = []
for datetime in datetimes:
   dates.append((str(datetime).split(' ')[0]))
max_day = df['Country'].value_counts().max()


new_df = []
countries = df['Country'].unique()
for country in countries:
    day_count = 1
    total_cases = 0
    total_deaths = 0
    df_c = df[df['Country']==country]
    no_data = True
    for date in dates:
        match = False
        for i in range(0,len(df_c['Date'])):
            if str(df_c['Date'].iloc[i]) == str(date):    
                new_df.append([df_c.iloc[i].tolist()[0],day_count,df_c.iloc[i].tolist()[7],df_c.iloc[i].tolist()[8],int(df_c.iloc[i].tolist()[4]),int(df_c.iloc[i].tolist()[5]),0,0]) #Date, DayNum, Code, Pop, Cases, Deaths, Total Cases, Total Deaths
                no_data = False
                match = True
                break
        if not match:
            if no_data:
                new_df.append([date,day_count,df_c.iloc[i].tolist()[7],df_c.iloc[i].tolist()[8],'No Data','No Data','No Data','No Data'])
            else:
                new_df.append([date,day_count,df_c.iloc[i].tolist()[7],df_c.iloc[i].tolist()[8],0,0,0,0])
        try:
            total_cases += new_df[-1][4]
            total_deaths += new_df[-1][5]
            new_df[-1][6] = total_cases
            new_df[-1][7] = total_deaths
        except:
            pass
        day_count += 1
        
df = pd.DataFrame(new_df, columns = ['Date','DayNum','Code','Pop','Cases','Deaths','Total Cases','Total Deaths'])

global_cases_list = [] # Global Stats
global_deaths_list = []
global_cases = 0
global_deaths = 0
for date in dates:
    df_date = df[df['Date'] == date]
    for country in list(df_date.Code.unique()):
        if country == country:
            cases = df_date.loc[df_date['Code'] == country, 'Cases'].iloc[0]
            deaths = df_date.loc[df_date['Code'] == country, 'Deaths'].iloc[0]
            if cases != 'No Data':
                global_cases += cases
            if deaths != 'No Data':
                global_deaths += deaths
    global_cases_list.append(global_cases)
    global_deaths_list.append(global_deaths)

# Setup map
high_cases = 100 # Define Highs
high_total_cases = 1000
high_deaths = 200
high_total_deaths = 100
show_stocks = True

shapename = 'admin_0_countries' # Get shape file
countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)

fig, (ax_m, ax_g_cases_cases, ax_g_cases_deaths) = plt.subplots(3) # Format Figures
plt.subplots_adjust(top=1,bottom=0, left=0, right=1)

cmap = plt.cm.get_cmap('gist_heat_r') # Colourmaps
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[int(len(cmaplist)/2):]
cmap = cmap.from_list('HalfHeat', cmaplist, cmap.N)

cmap_countries = plt.cm.get_cmap('Reds')
cmap_deaths = cmap.from_list('HalfHeat', cmaplist, cmap.N)
cmap_indicies = plt.cm.get_cmap('Reds')

ax_m = plt.axes(projection=ccrs.PlateCarree()) # Map

ax_s = plt.axes([0.25, 0.97, 0.5, 0.015], facecolor='lightgrey') # Slider
slider = Slider(ax_s, 'Date', 1, max_day, valinit=max_day, valstep=1, valfmt = dates[max_day-1],fc='grey')

'''
ax_c  = fig.add_axes([0.25,0.2,0.5,0.03]) # Colour Bar
norm = mpl.colors.Normalize(vmin=0,vmax=high_cases)

cb  = mpl.colorbar.ColorbarBase(ax_c,cmap=cmap_countries,norm=norm,ticks=[x for x in range(0,high_total_cases+100,100)],orientation='horizontal')
ticks = [x for x in range(0,high_total_cases,100)]
ticks.append('>'+str(high_total_cases))
cb.ax.set_xticklabels(ticks)
cb.set_label('Total Cases')
'''

ax_i  = fig.add_axes([0.155,-0.15,0.5,0.5]) # Colour Matrix
def darken(colour, factor):
    r, g, b, a = colour[0], colour[1], colour[2], colour[3]
    h, l, s = rgb_to_hls(r, g, b)
    l = max(min(l * factor, 1), 0)
    r, g, b = hls_to_rgb(h, l, s)
    return (r, g, b, a)
    
cmap_mat = plt.cm.get_cmap('OrRd')
cmap_mat_list = [cmap_mat(i) for i in range(cmap_mat.N)]
cmap_mat_list = cmap_mat_list[:int(len(cmap_mat_list)*0.8)]
cmap_mat = cmap.from_list('3/4OrRd', cmap_mat_list, cmap.N)
new_row = [cmap_mat(i) for i in range(cmap.N)]
mat = []
for i in range(0,25):
    mat.append(new_row)
    new_row = []
    for c in mat[-1]:
        f = (1 - 0.08**(1+((i/25)*0.3)))
        new_row.append(darken(c,f))
mat = np.array(mat)
mat = np.flip(mat,0)

def colour_matrix(scaled_x, scaled_y):
    rows, cols = mat.shape[0], mat.shape[1]
    return mat[(int(scaled_y*rows)-1)*(int(scaled_y*rows)>0)][(int(scaled_x*cols)-1)*(int(scaled_x*cols)>0)]

ax_i = plt.imshow(mat, interpolation='bilinear')       
plt.setp(plt.title('Country Colours:'), size=15,color='black',backgroundcolor='None') 
tick_labels = [str(i) for i in range(0,high_total_cases,int(high_total_cases/(5-1)))]
tick_labels.append('>'+str(high_total_cases))
ticks = list(np.linspace(0,mat.shape[1]-1,5))
plt.xticks(ticks, tick_labels)
plt.xlabel(text='Total Cases',size=13,color='black',backgroundcolor='None',xlabel=None)
ticks = [0,mat.shape[0]-1]
tick_labels = ['>'+str(high_deaths),0]
plt.yticks(ticks, tick_labels)
plt.ylabel(text='Deaths\nToday',size=13,color='black',backgroundcolor='None',rotation=0,ylabel=None,position=(0,0.35))

# Display
def update(val):
    # Figure Updates
    ax_m.cla() # Map Update
    ax_m.stock_img()
    #ax_m.background_img(name='ne_shaded', resolution='high')
    ax_m.add_patch(patches.Rectangle((-141,-90),279,33,fc='gray',alpha=0.9,ec='black'))

    date = dates[int(val-1)]
    slider.valtext.set_text(dates[int(val-1)]) # Slider Update
    DayNum = int(slider.val)

    graphs_face_colour = (0.9,0.9,0.9)
    ax_g_cases  = fig.add_axes([0.68,0.047,0.06,0.1]) # Global Cases Graph Update
    ax_g_cases.cla()
    plt.xticks([0,len(dates)], ['01/01/2020',dates[-1]])
    ax_g_cases.get_xaxis().set_visible(False)
    ax_g_cases.get_yaxis().set_visible(False)
    ax_g_cases.set_facecolor(graphs_face_colour)
    #ax_g_cases.set_title(text='Global Cases',color='black',backgroundcolor='white',position=(.5, 1.02),label=None) 
    ax_g_cases = plt.plot([i for i in range(0,DayNum)],global_cases_list[0:DayNum],c='red',marker='None',mfc='red',mec='none',ls='-',lw=1)

    ax_g_deaths  = fig.add_axes([0.786,0.047,0.06,0.1]) # Global Deaths Graph Update
    ax_g_deaths.cla()
    plt.xticks([0,len(dates)], ['01/01/2020',dates[-1]])
    ax_g_deaths.get_xaxis().set_visible(False)
    ax_g_deaths.get_yaxis().set_visible(False)
    ax_g_deaths.set_facecolor(graphs_face_colour)
    #ax_g_deaths.set_title(text='Global Cases',color='black',backgroundcolor='white',position=(.5, 1.02),label=None) 
    ax_g_deaths = plt.plot([i for i in range(0,DayNum)],global_deaths_list[0:DayNum],c=(77/255, 0, 0),marker='None',mfc='red',mec='none',ls='-',lw=1)
    
    
    # Data Updates
    df_day = df[df['DayNum'] == DayNum]
    global_cases = 0
    global_deaths = 0
    # Country Data Update
    for country in shpreader.Reader(countries_shp).records(): # Country Info       
        code = country.attributes['ADM0_A3']
        name = country.attributes['NAME_LONG']
        try:
            lat = df_coords.loc[df_coords['Code'] == code, 'Lat'].iloc[0]
            long = df_coords.loc[df_coords['Code'] == code, 'Long'].iloc[0]
            population = df_day.loc[df_day['Code'] == code, 'Pop'].iloc[0]
            cases = df_day.loc[df_day['Code'] == code, 'Cases'].iloc[0]
            total_cases = df_day.loc[df_day['Code'] == code, 'Total Cases'].iloc[0]
            deaths = df_day.loc[df_day['Code'] == code, 'Deaths'].iloc[0]
            total_deaths = df_day.loc[df_day['Code'] == code, 'Total Deaths'].iloc[0]
            global_cases += total_cases
            global_deaths += total_deaths
            #ax_m.text(long, lat, name, fontsize=6);
            
            # Country Masks
            if total_cases == 0 : # Borders - Total Cases
                edge_colour = 'black'
            else:
                edge_colour = 'red'
            #scaled = (total_deaths / total_cases)*((total_deaths / total_cases)<=high_deathratio) + ((total_deaths / total_cases)>high_deathratio)
            scaled_cases = 0.5 + ((cases / high_cases)*(cases<=high_cases) + (cases>high_cases))*0.5
            width = scaled_cases * 1.5

            scaled_total_cases = (total_cases/high_total_cases)*(total_cases<=high_total_cases) + (total_cases>high_total_cases) # Face - Total Cases , Deaths
            scaled_deaths = (deaths / high_deaths)*(deaths<=high_deaths) + (deaths>high_deaths)
            face_colour = colour_matrix(scaled_total_cases, 1-scaled_deaths)
            #alpha = (0.1 + (total_deaths / total_cases)*10)*((total_deaths / total_cases)*10 <= 0.9) + ((total_deaths / total_cases)*10 > 0.9)
            ax_m.add_geometries(country.geometry, ccrs.PlateCarree(),fc=face_colour, ec=edge_colour, linewidth=width, label=name, alpha=0.75)     

            # Country Bubbles
            '''
            if deaths == 0 :
                size = 0
            else:
                size = 5 + (scaled**0.33)*10           
            face_colour = cmap_deaths(scaled)
            edge_color = 'None'
            ax_m.plot(long, lat ,marker='D',ms=size, c=face_colour,alpha=0.8,mew=1,mec=edge_colour)
            '''
            
        except IndexError: # Missing Data
            face_colour = (0.7,0.7,0.7,0.6)
            edge_colour = (0.3,0.3,0.3,1)
            ax_m.add_geometries(country.geometry, ccrs.PlateCarree(),fc=face_colour, ec = edge_colour, label=name)
            print("{} : Missing Data".format(name))

        except: # No Data
            face_colour = (0.8,0.8,0.8,0.6)
            edge_colour = (0.3,0.3,0.3,1)
            ax_m.add_geometries(country.geometry, ccrs.PlateCarree(),fc=face_colour, ec = edge_colour, label=name)
            #print("{} : No Data".format(name))
            
    # National Indicies Data Update
    for ind in nat_indicies :
        date = dates[DayNum-1] # Index Info
        df_ind = ind_dfs[ind]
        initial_price = df_ind['Close'].iloc[0]
        lat = nat_indicies[ind][1][0]
        long = nat_indicies[ind][1][1]
        name = nat_indicies[ind][0]
        try:
            price = df_ind.loc[df_ind['Date'] == date,'Close'].iloc[0]
            if DayNum != 1:
                yesturday_price = df_ind.loc[df_ind['Date'] == dates[DayNum-2],'Close'].iloc[0]
            else:
                yesturday_price = price
        except IndexError: # Missing Data
            price = df_ind['Close'].iloc[-1]
            yesturday_price = df_ind['Close'].iloc[-2] 
        total_perc_change = ((price - initial_price) / initial_price)
        daily_perc_change = ((price - yesturday_price) / yesturday_price) 

        size = abs(total_perc_change) * 100 # Index Bubbles
        if total_perc_change > 0 :
            face_colour = 'lime'
        else:
            #face_colour = cmap_indicies(abs(total_perc_change)+20)
            face_colour = (205/255, 0, 0)
        if daily_perc_change > 0 :
            edge_colour = 'lime'
        else:
            edge_colour = 'red'

        if show_stocks:
            ax_m.plot(long, lat ,marker='o',ms=size, c=face_colour,alpha=1,mew=1,mec=edge_colour)
            ax_m.text(long+8, lat-(name=='SSE Composite Index')*1.5, name, fontsize=10, c = 'black', bbox=dict(boxstyle="square",fc='white',ec='None'))
            ax_m.text(long+7.8, lat-2.1-(name=='SSE Composite Index')*1.55, '{0:.1f}%'.format(total_perc_change*100), fontsize=8, c = face_colour, bbox=dict(boxstyle="square",fc='white',ec='None'))
            ax_m.text(long+16, lat-1.82-(name=='SSE Composite Index')*1.55, '{0:.1f}%'.format(daily_perc_change*100), fontsize=6, c = edge_colour, bbox=dict(boxstyle="square",fc='white',ec='None'))
        
    months = ['Jan','Feb','Mar','April','May','June','July','Aug','Sep','Oct','Nov','Dec'] # Text Boxes
    datel = date.split('/')
    long_date = "{} {} {}".format(datel[0],months[int(datel[1])-1],datel[2])
    
    ax_m.text(-16, -54.45, "{}".format(long_date), color='black', fontsize=20, bbox=dict(boxstyle="square",fc='gray',ec='black'))
    ax_m.text(64, -88, "Global Cases:          \n\n\n\n\n\n{}".format(global_cases), color='black',fontsize=15, bbox=dict(boxstyle="square",fc='None',ec='None'))   
    ax_m.text(102, -88, "Global Deaths:         \n\n\n\n\n\n{}".format(global_deaths), color='black',fontsize=15, bbox=dict(boxstyle="square",fc='None',ec='None'))
    
# Show
update(max_day)
slider.on_changed(update)
plt.show()















