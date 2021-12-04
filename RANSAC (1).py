from sklearn import linear_model
import pandas as pd
import numpy as np
import quandl as quandl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplcursors as mplcursors


### Import historical bitcoin price from quandl
df = quandl.get("BCHAIN/MKPRU", api_key="FYzyusVT61Y4w65nFESX").reset_index()
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by="Date", inplace=True)
df = df[df["Value"] > 0]

### RANSAC Regression
def LinearReg(ind, value):
    X = np.array(np.log(ind)).reshape(-1, 1)
    y = np.array(np.log(value))
    ransac = linear_model.RANSACRegressor(residual_threshold=2.989, random_state=0)
    ransac.fit(X, y)
    LinearRegRANSAC = ransac.predict(X)
    return LinearRegRANSAC

df["LinearRegRANSAC"] = LinearReg(df.index, df.Value)

#### Plot
fig = make_subplots()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Value"], name="Price", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=df["Date"], y=np.exp(df["LinearRegRANSAC"]), name="Ransac", line=dict(color="green")))
fig.update_layout(template="plotly_dark")
mplcursors.cursor(hover=True)
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Price", type='log', showgrid=True)
fig.show()

