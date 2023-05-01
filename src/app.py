
import pathlib
import numpy as np # para usar el logaritmo n
import pandas as pd
import yfinance as yf
import math as math
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import openpyxl



PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('../data').resolve()

litickers = ['T', 'TROW', 'IRM', 'TRTN']


lidfofline = [pd.read_excel(DATA_PATH.joinpath('df{}.xlsx'.format(litickers[x])), index_col=0) for x in
              range(len(litickers))]


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])
server = app.server

dftp = pd.DataFrame(list(zip(litickers, lidfofline)),
                    columns=['ticker', 'dft'])

# lioptionspesos = [{'label':str(option),'value':option} for option in lipesos]
lioptionstkr = [{'label': str(option), 'value': option} for option in litickers]

carinput = dbc.Card(
    [
        dbc.CardBody([
            html.H4("SELECTOR DE TICKERS", className="card-title"),
            html.H6("INTRODUCE CAPITAL A INVERTIR:", className="card-text"),
            dcc.Input(id='micapital', type='number', placeholder='capital inicial', min=100000, max=1000000),
            html.Hr(),
            html.H6("SELECCIONA TICKER ", className="card-text"),
            dcc.Dropdown(id='ticker-picker', options=lioptionstkr, value=litickers, multi=True),
            html.Hr(),
            html.H6("SELECCIONA T.E.R ", className="card-text"),
            dcc.Input(id='miter', type='number', placeholder='capital inicial', min=0.07, max=0.2, step=0.01),
            html.Hr(),
            html.Button(id="boton1", n_clicks=0, children="calcular")
        ]),
    ], color="#d5def5",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(carinput, width=4)
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dcc.Loading(children=[html.Div(id='caroutput', children=[])]), width=12)
            ]
        )
    ]
)


@app.callback(Output('caroutput', 'children'),
              Input('boton1', 'n_clicks'),
              State('ticker-picker', 'value'),
              State('micapital', 'value'),
              State('miter', 'value')
              )
def rendimientocarteras(n, liticker, capital, ter):
    dff = dftp[dftp.ticker.isin(liticker)]

    listadf = list(dff['dft'])

    lipesos = [1 / (len(listadf)) for x in range(len(listadf))]
    epacopiota = [x * capital for x in lipesos]

    listadfvalfin = []

    for x in range(len(listadf)):

        dfi = listadf[x]
        dfetfdivi = dfi[['Close', 'Dividends']].copy()
        dfetfdivi.rename(columns={'Dividends': 'divi'}, inplace=True)

        dfetfdivi['ratebh'] = (dfetfdivi['Close'].pct_change(periods=1))
        dfetfdivi['sesion'] = np.arange(len(dfetfdivi))  # numerar sesiones
        dfetfdivi['Date'] = dfetfdivi.index  # para pasar el indice de fechas a columna
        coldate = dfetfdivi.pop('Date')
        dfetfdivi.insert(0, 'Date', coldate)
        dfetfdivi = dfetfdivi.set_index('sesion')
        dfetfdivi['year'] = pd.to_datetime(dfetfdivi['Date']).dt.year
        dfetfdivi['cambioyear'] = dfetfdivi['year'].diff()
        dfetfdivi['mes'] = pd.to_datetime(dfetfdivi['Date']).dt.month
        dfetfdivi['mesrecu'] = np.where(dfetfdivi['mes'] == 6, 1, 0)
        dfetfdivi['diarecu'] = dfetfdivi['mesrecu'].diff()
        dfetfdivi['diarecu'] = dfetfdivi['diarecu'].replace([-1], 0)
        dfetfdivi['valfinacc'] = list(dfetfdivi['Close'])[-1]

        dfetfdivi.loc[0, 'valin'] = epacopiota[x]
        dfetfdivi.loc[0, 'Ncomp'] = 0
        dfetfdivi.loc[0, 'Nacu'] = 0
        dfetfdivi.loc[0, 'cash'] = epacopiota[x]
        dfetfdivi.loc[0, 'divinet'] = 0
        dfetfdivi.loc[0, 'retenor'] = 0
        dfetfdivi.loc[0, 'divibrut'] = 0
        dfetfdivi.loc[0, 'retenoracu'] = 0
        dfetfdivi.loc[0, 'retefinyear'] = 0
        dfetfdivi.loc[0, 'retefinyearacu'] = 0
        dfetfdivi.loc[0, 'pastarecu'] = 0
        dfetfdivi.loc[0, 'retespainacu'] = 0
        dfetfdivi.loc[0, 'retespainfinyearacu'] = 0
        dfetfdivi.loc[0, 'retespainfinyear'] = 0
        dfetfdivi.loc[0, 'pastadebo'] = 0

        dfetfdivi.fillna(0)

        for i in np.arange(1, len(dfetfdivi)):

            if dfetfdivi.loc[i - 1, 'cash'] > dfetfdivi.loc[i, 'Close']:
                dfetfdivi.loc[i, 'Ncomp'] = math.floor(dfetfdivi.loc[i - 1, 'cash'] / dfetfdivi.loc[i, 'Close'])
            else:
                dfetfdivi.loc[i, 'Ncomp'] = 0
            dfetfdivi.loc[i, 'Nacu'] = dfetfdivi.loc[i, 'Ncomp'] + dfetfdivi.loc[i - 1, 'Nacu']
            dfetfdivi.loc[i, 'divibrut'] = dfetfdivi.loc[i, 'Nacu'] * dfetfdivi.loc[i, 'divi']
            dfetfdivi.loc[i, 'retenor'] = dfetfdivi.loc[i, 'divibrut'] * .15
            dfetfdivi.loc[i, 'divinet'] = (dfetfdivi.loc[i, 'divibrut'] - dfetfdivi.loc[i, 'retenor'])
            dfetfdivi.loc[i, 'retespain'] = dfetfdivi.loc[i, 'divinet'] * 0.19
            dfetfdivi.loc[i, 'retespainacu'] = dfetfdivi.loc[i, 'retespain'] + dfetfdivi.loc[i - 1, 'retespainacu'] - \
                                               dfetfdivi.loc[i - 1, 'retespainfinyear']
            dfetfdivi.loc[i, 'retenoracu'] = dfetfdivi.loc[i, 'retenor'] + dfetfdivi.loc[i - 1, 'retenoracu'] - \
                                             dfetfdivi.loc[i - 1, 'retefinyear']
            dfetfdivi.loc[i, 'retespainfinyear'] = dfetfdivi.loc[i, 'retespainacu'] * dfetfdivi.loc[i, 'cambioyear']
            dfetfdivi.loc[i, 'retefinyear'] = dfetfdivi.loc[i, 'retenoracu'] * dfetfdivi.loc[i, 'cambioyear']
            dfetfdivi.loc[i, 'retespainfinyearacu'] = dfetfdivi.loc[i, 'retespainfinyear'] + dfetfdivi.loc[
                i - 1, 'retespainfinyearacu'] - dfetfdivi.loc[i - 1, 'pastadebo']
            dfetfdivi.loc[i, 'retefinyearacu'] = dfetfdivi.loc[i, 'retefinyear'] + dfetfdivi.loc[
                i - 1, 'retefinyearacu'] - dfetfdivi.loc[i - 1, 'pastarecu']
            dfetfdivi.loc[i, 'pastadebo'] = dfetfdivi.loc[i, 'diarecu'] * dfetfdivi.loc[i, 'retespainfinyearacu']
            dfetfdivi.loc[i, 'pastarecu'] = dfetfdivi.loc[i, 'diarecu'] * dfetfdivi.loc[i, 'retefinyearacu']
            dfetfdivi.loc[i, 'cash'] = dfetfdivi.loc[i - 1, 'cash'] + dfetfdivi.loc[i, 'divinet'] - dfetfdivi.loc[
                i, 'Ncomp'] * dfetfdivi.loc[i, 'Close'] + dfetfdivi.loc[i, 'pastarecu'] - dfetfdivi.loc[i, 'pastadebo']
            dfetfdivi.loc[i, 'valfin'] = dfetfdivi.loc[i, 'Nacu'] * dfetfdivi.loc[i, 'Close']

        dfetfdivi['cantcompra'] = dfetfdivi['Ncomp'] * dfetfdivi['Close']
        dfetfdivi['cantcompraacu'] = dfetfdivi['cantcompra'].cumsum()
        dfetfdivi['ganabrut'] = dfetfdivi['valfin'] - dfetfdivi['cantcompraacu']

        filters = [(dfetfdivi['ganabrut'] <= 0), (dfetfdivi['ganabrut'] > 0) & (dfetfdivi['ganabrut'] <= 6000)
            , (dfetfdivi['ganabrut'] > 6000) & (dfetfdivi['ganabrut'] <= 50000),
                   (dfetfdivi['ganabrut'] > 50000) & (dfetfdivi['ganabrut'] <= 200000),
                   (dfetfdivi['ganabrut'] > 200000)]
        values = [0, 0.19, 0.21, 0.22, 0.26]

        dfetfdivi["tasa"] = np.select(filters, values)

        dfetfdivi['impuestosfinal'] = dfetfdivi['ganabrut'] * dfetfdivi["tasa"]
        dfetfdivi['ganancianetadivi'] = dfetfdivi['valfin'] + dfetfdivi['cash'] - dfetfdivi['impuestosfinal'] - \
                                        epacopiota[x]

        listavalfin = list(dfetfdivi['ganancianetadivi'])
        listadfvalfin.append(listavalfin)

    for j in range(len(listadfvalfin)):
        dfetfdivi[f'ganancianetadivi {j}'] = listadfvalfin[j]

    dfetfdivi['total'] = dfetfdivi.iloc[:, -len(listadfvalfin):].sum(axis=1)

    ganetodivifin = list(dfetfdivi['total'])[-1]
    ganetodivifinf = "{:,.0f}".format(ganetodivifin)

    listadfvalfinfon = []

    # dfetfdivi.to_excel("divideando5.xlsx")

    for x in range(len(listadf)):

        dfi = listadf[x]
        dfetffon = dfi[['Close', 'Dividends']].copy()
        dfetffon.rename(columns={'Dividends': 'divi'}, inplace=True)

        dfetffon['ratebh'] = (dfetffon['Close'].pct_change(periods=1))
        dfetffon['sesion'] = np.arange(len(dfetffon))  # numerar sesiones
        dfetffon['Date'] = dfetffon.index  # para pasar el indice de fechas a columna
        coldate = dfetffon.pop('Date')
        dfetffon.insert(0, 'Date', coldate)
        dfetffon = dfetffon.set_index('sesion')
        dfetffon['year'] = pd.to_datetime(dfetffon['Date']).dt.year
        dfetffon['cambioyear'] = dfetffon['year'].diff()
        dfetffon['mes'] = pd.to_datetime(dfetffon['Date']).dt.month
        dfetffon['mesrecu'] = np.where(dfetffon['mes'] == 6, 1, 0)
        dfetffon['diarecu'] = dfetffon['mesrecu'].diff()
        dfetffon['diarecu'] = dfetffon['diarecu'].replace([-1], 0)
        dfetffon['valfinacc'] = list(dfetffon['Close'])[-1]

        dfetffon.loc[0, 'valin'] = epacopiota[x]
        dfetffon.loc[0, 'Ncomp'] = 0
        dfetffon.loc[0, 'Nacu'] = 0
        dfetffon.loc[0, 'cash'] = epacopiota[x]
        dfetffon.loc[0, 'divinet'] = 0
        dfetffon.loc[0, 'retenor'] = 0
        dfetffon.loc[0, 'divibrut'] = 0
        dfetffon.loc[0, 'retenoracu'] = 0
        dfetffon.loc[0, 'retefinyear'] = 0
        dfetffon.loc[0, 'retefinyearacu'] = 0
        dfetffon.loc[0, 'pastarecu'] = 0
        dfetffon.loc[0, 'retespainacu'] = 0
        dfetffon.loc[0, 'retespainfinyearacu'] = 0
        dfetffon.loc[0, 'retespainfinyear'] = 0
        dfetffon.loc[0, 'pastadebo'] = 0
        dfetffon.loc[0, 'ter'] = 0

        dfetffon.fillna(0)

        for i in np.arange(1, len(dfetffon)):

            if dfetffon.loc[i - 1, 'cash'] > dfetffon.loc[i, 'Close']:
                dfetffon.loc[i, 'Ncomp'] = math.floor(dfetffon.loc[i - 1, 'cash'] / dfetffon.loc[i, 'Close'])
            else:
                dfetffon.loc[i, 'Ncomp'] = 0
            dfetffon.loc[i, 'Nacu'] = dfetffon.loc[i, 'Ncomp'] + dfetffon.loc[i - 1, 'Nacu']
            dfetffon.loc[i, 'divibrut'] = dfetffon.loc[i, 'Nacu'] * dfetffon.loc[i, 'divi']
            dfetffon.loc[i, 'retenor'] = dfetffon.loc[i, 'divibrut'] * .15
            dfetffon.loc[i, 'divinet'] = (dfetffon.loc[i, 'divibrut'] - dfetffon.loc[i, 'retenor'])
            dfetffon.loc[i, 'retespain'] = dfetffon.loc[i, 'divinet'] * 0
            dfetffon.loc[i, 'retespainacu'] = dfetffon.loc[i, 'retespain'] + dfetffon.loc[i - 1, 'retespainacu'] - \
                                              dfetffon.loc[i - 1, 'retespainfinyear']
            dfetffon.loc[i, 'retenoracu'] = dfetffon.loc[i, 'retenor'] + dfetffon.loc[i - 1, 'retenoracu'] - \
                                            dfetffon.loc[i - 1, 'retefinyear']
            dfetffon.loc[i, 'retespainfinyear'] = dfetffon.loc[i, 'retespainacu'] * dfetffon.loc[i, 'cambioyear']
            dfetffon.loc[i, 'retefinyear'] = dfetffon.loc[i, 'retenoracu'] * dfetffon.loc[i, 'cambioyear']
            dfetffon.loc[i, 'retespainfinyearacu'] = dfetffon.loc[i, 'retespainfinyear'] + dfetffon.loc[
                i - 1, 'retespainfinyearacu'] - dfetffon.loc[i - 1, 'pastadebo']
            dfetffon.loc[i, 'retefinyearacu'] = dfetffon.loc[i, 'retefinyear'] + dfetffon.loc[i - 1, 'retefinyearacu'] - \
                                                dfetffon.loc[i - 1, 'pastarecu']
            dfetffon.loc[i, 'pastadebo'] = dfetffon.loc[i, 'diarecu'] * dfetffon.loc[i, 'retespainfinyearacu']
            dfetffon.loc[i, 'pastarecu'] = dfetffon.loc[i, 'diarecu'] * dfetffon.loc[i, 'retefinyearacu']
            dfetffon.loc[i, 'cash'] = dfetffon.loc[i - 1, 'cash'] + dfetffon.loc[i, 'divinet'] - dfetffon.loc[
                i, 'Ncomp'] * dfetffon.loc[i, 'Close']
            dfetffon.loc[i, 'valfin'] = dfetffon.loc[i, 'Nacu'] * dfetffon.loc[i, 'Close']
            dfetffon.loc[i, 'ter'] = (dfetffon.loc[i, 'valfin'] * ter / 100) * i / 252

        dfetffon['cantcompra'] = dfetffon['Ncomp'] * dfetffon['Close']
        dfetffon['cantcompraacu'] = dfetffon['cantcompra'].cumsum()
        dfetffon['ganabrut'] = dfetffon['valfin'] - dfetffon['cantcompraacu'] - dfetffon['ter']

        filters = [(dfetffon['ganabrut'] <= 0), (dfetffon['ganabrut'] > 0) & (dfetffon['ganabrut'] <= 6000)
            , (dfetffon['ganabrut'] > 6000) & (dfetffon['ganabrut'] <= 50000),
                   (dfetffon['ganabrut'] > 50000) & (dfetffon['ganabrut'] <= 200000), (dfetffon['ganabrut'] > 200000)]
        values = [0, 0.19, 0.21, 0.22, 0.26]

        dfetffon["tasa"] = np.select(filters, values)

        dfetffon['impuestosfinal'] = dfetffon['ganabrut'] * dfetffon["tasa"]
        dfetffon['ganancianetadivi'] = dfetffon['valfin'] - dfetffon['ter'] + dfetffon['cash'] - dfetffon[
            'impuestosfinal'] - epacopiota[x]

        listavalfinfon = list(dfetffon['ganancianetadivi'])
        listadfvalfinfon.append(listavalfinfon)

    for j in range(len(listadfvalfinfon)):
        dfetffon[f'ganancianetadivi {j}'] = listadfvalfinfon[j]

    dfetffon['total'] = dfetffon.iloc[:, -len(listadfvalfinfon):].sum(axis=1)

    ganetofonfin = list(dfetffon['total'])[-1]
    ganetofonfinf = "{:,.0f}".format(ganetofonfin)
    porcentaje = (ganetofonfin - ganetodivifin) / ganetofonfin * 100
    porcentajef = "{:,.2f}".format(porcentaje)

    traces = [
        go.Scatter(
            x=list(dfetfdivi['Date']),
            y=dfetfdivi['total'],
            mode='lines',
            text='dividendero',
            name='ganancia neta dividendero'),

        go.Scatter(
            x=list(dfetffon['Date']),
            y=dfetffon['total'],
            mode='lines',
            text='fondo',
            name='ganancia neta fondo'),
    ]
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(traces[0], secondary_y=False)
    fig1.add_trace(traces[1], secondary_y=False)

    caroutput = dbc.Card(
        [
            dbc.CardBody([
                html.H4("COMPARATIVA AGREGADA EVOLUCIÓN GANANCIA NETA DIVIDENDERO/FONDO", className="card-title"),
                dbc.Row(
                    [dbc.Col(dcc.Graph(figure=fig1), width=12)
                     ]
                ),
                html.H4("GANACIA NETA AL FINAL DIVIDENDER0: {} $".format(ganetodivifinf), className="card-title"),
                html.H4("GANACIA NETA AL FINAL FONDO: {} $".format(ganetofonfinf), className="card-title"),
                html.H4("% PERDIDA FONDO VS DIVIDENDERO: {} %".format(porcentajef), className="card-title"),
                # html.H4("COMPARATIVA EVOLUCIÓN GANANCIA NETA DIVIDENDERO", className="card-title"),
                # dbc.Row(
                #     [dbc.Col(dcc.Graph(figure= fig2),width=12)
                #      ]
                # ),
            ]),
        ], color="#d5def5", inverse=False
    )

    return caroutput


if __name__ == '__main__':
    app.run_server()



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server

sidebar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className="bg-light",
)

app.layout = dbc.Container([
    dcc.Store(id="mydata_stored", data=dimydata),
    dbc.Row([
        dbc.Col(html.Div("ULTRABASIC APP DATA SHARED",
                         style={'fontSize': 50, 'textAlign': 'center'}))
    ]),
    html.Hr(),
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),
            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)

if __name__ == "__main__":
    app.run(debug=False)
