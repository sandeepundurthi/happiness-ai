import pandas as pd
import joblib
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Load data and model
df = pd.read_csv("data/processed/happiness_clean.csv")
model = joblib.load("models/happiness_xgb.pkl")

features = [
    "explained_log_gdp_per_capita",
    "explained_social_support",
    "explained_healthy_life_expectancy",
    "explained_freedom",
    "explained_generosity",
    "explained_corruption",
    "dystopia_plus_residual"
]

latest_year = df["year"].max()

app = Dash(__name__)
app.title = "HappiScope AI"

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "25px", "backgroundColor": "#f7f9fc"},
    children=[
        html.H1("HappiScope AI: Global Happiness Intelligence", style={"textAlign": "center"}),

        html.P(
            "Analyze global happiness trends, compare countries, and predict happiness scores using socioeconomic indicators.",
            style={"textAlign": "center", "fontSize": "18px"}
        ),

        html.Div([
            html.Label("Select Country:"),
            dcc.Dropdown(
                id="country-dropdown",
                options=[{"label": c, "value": c} for c in sorted(df["country"].unique())],
                value="United States",
                clearable=False
            )
        ], style={"width": "45%", "margin": "auto"}),

        html.Br(),

        dcc.Tabs([
            dcc.Tab(label="Global Trend", children=[
                dcc.Graph(id="global-trend")
            ]),

            dcc.Tab(label="Country Explorer", children=[
                dcc.Graph(id="country-trend")
            ]),

            dcc.Tab(label="Top Countries", children=[
                dcc.Graph(id="top-countries")
            ]),

            dcc.Tab(label="Feature Importance", children=[
                dcc.Graph(id="feature-importance")
            ]),

            dcc.Tab(label="Predict Happiness", children=[
                html.Div(style={"padding": "25px"}, children=[
                    html.H3("Predict Happiness Score"),

                    html.Label("GDP per Capita Contribution"),
                    dcc.Slider(0, 2, 0.01, value=1.2, id="gdp-slider"),

                    html.Label("Social Support Contribution"),
                    dcc.Slider(0, 2, 0.01, value=1.2, id="support-slider"),

                    html.Label("Healthy Life Expectancy Contribution"),
                    dcc.Slider(0, 2, 0.01, value=0.8, id="health-slider"),

                    html.Label("Freedom Contribution"),
                    dcc.Slider(0, 1, 0.01, value=0.5, id="freedom-slider"),

                    html.Label("Generosity Contribution"),
                    dcc.Slider(0, 1, 0.01, value=0.2, id="generosity-slider"),

                    html.Label("Corruption Contribution"),
                    dcc.Slider(0, 1, 0.01, value=0.2, id="corruption-slider"),

                    html.Label("Dystopia + Residual"),
                    dcc.Slider(0, 4, 0.01, value=2.0, id="residual-slider"),

                    html.Br(),
                    html.H2(id="prediction-output", style={"textAlign": "center"})
                ])
            ])
        ])
    ]
)


@app.callback(
    Output("global-trend", "figure"),
    Input("country-dropdown", "value")
)
def update_global_trend(country):
    yearly = df.groupby("year", as_index=False)["happiness_score"].mean()

    fig = px.line(
        yearly,
        x="year",
        y="happiness_score",
        markers=True,
        title="Global Average Happiness Score Over Time"
    )
    fig.update_layout(template="plotly_white")
    return fig


@app.callback(
    Output("country-trend", "figure"),
    Input("country-dropdown", "value")
)
def update_country_trend(country):
    country_df = df[df["country"] == country]

    fig = px.line(
        country_df,
        x="year",
        y="happiness_score",
        markers=True,
        title=f"{country} Happiness Trend"
    )
    fig.update_layout(template="plotly_white")
    return fig


@app.callback(
    Output("top-countries", "figure"),
    Input("country-dropdown", "value")
)
def update_top_countries(country):
    latest = (
        df[df["year"] == latest_year]
        .sort_values("happiness_score", ascending=False)
        .head(15)
    )

    fig = px.bar(
        latest,
        x="happiness_score",
        y="country",
        orientation="h",
        title=f"Top 15 Happiest Countries ({latest_year})"
    )
    fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
    return fig


@app.callback(
    Output("feature-importance", "figure"),
    Input("country-dropdown", "value")
)
def update_feature_importance(country):
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Model Feature Importance"
    )
    fig.update_layout(template="plotly_white")
    return fig


@app.callback(
    Output("prediction-output", "children"),
    [
        Input("gdp-slider", "value"),
        Input("support-slider", "value"),
        Input("health-slider", "value"),
        Input("freedom-slider", "value"),
        Input("generosity-slider", "value"),
        Input("corruption-slider", "value"),
        Input("residual-slider", "value"),
    ]
)
def predict_happiness(gdp, support, health, freedom, generosity, corruption, residual):
    input_df = pd.DataFrame([{
        "explained_log_gdp_per_capita": gdp,
        "explained_social_support": support,
        "explained_healthy_life_expectancy": health,
        "explained_freedom": freedom,
        "explained_generosity": generosity,
        "explained_corruption": corruption,
        "dystopia_plus_residual": residual
    }])

    prediction = model.predict(input_df)[0]

    return f"Predicted Happiness Score: {prediction:.2f}"


if __name__ == "__main__":
    app.run(debug=True)
