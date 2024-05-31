from flask import Flask, jsonify, render_template, request, redirect, url_for
from datapipeV2 import DataProcessor
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__,static_folder='static')

df = pd.read_csv('data/processed_datav2.csv')
df.set_index('Date', inplace=True)
dft = df.tail(60)

processor = DataProcessor()

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        accuracy = None
        return render_template('predict.html', accuracy=accuracy)
    
    else:
        try:
            direction, predictions, accuracy, disdate = processor.update_data()
            predictions_html = predictions.to_html(classes='table table-bordered table-striped text-center', index=False)
            return render_template('predict.html', predictions=predictions_html, direction=direction, accuracy=accuracy, dis_date=disdate)
        except Exception as e:
            return redirect(url_for('error'))
        
@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/price', methods=['GET'])
def price():
    if request.method=='GET':

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=dft.index, y=dft['btc_price'], name='Bitcoin Price', mode='lines', line=dict(shape='spline',smoothing=1, color='green')))
        fig.update_traces(hoverinfo='x+y', hoverlabel=dict(bgcolor='green', bordercolor='white', font=dict(size=12)))
        fig.update_layout(xaxis=dict(type='date', showgrid=False),plot_bgcolor='white', paper_bgcolor='white',hovermode='x')

        fig.add_trace(go.Scatter(x=dft.index, y=dft['btc_rsi'], name='RSI 14d', mode='lines', line=dict(shape='spline', smoothing=1, color='red', width=1), 
                                 yaxis='y2', opacity=0.3 ))
        fig.update_layout(yaxis2=dict( overlaying='y', side='right', range=[0, 100]))

        plot_html = pio.to_html(fig, full_html=False)

        
        fig2 = go.Figure(layout=dict(barcornerradius=50))
        fig2.add_trace(go.Bar(
            x=dft.index,
            y=dft['btc_change'].apply(lambda x: x if x > 0 else 0),
            showlegend=False,
            marker_color='green',
            hoverinfo='x+y',
            
        ))
        fig2.add_trace(go.Bar(
            x=dft.index,
            y=dft['btc_change'].apply(lambda x: x if x < 0 else 0),
            showlegend=False,
            marker_color='red',
            hoverinfo='x+y',
        ))
        fig2.update_layout(
            xaxis=dict(type='date'),
            yaxis=dict(title='Price Change (%)'),
            plot_bgcolor='white',
            hovermode='x',
            barmode='relative',  
            showlegend=False
        )

        plot_html2 = pio.to_html(fig2, full_html=False)


        return render_template('price.html', plot=plot_html, plot2=plot_html2)
    

@app.route('/models', methods=['GET'])
def models():
    if request.method=='GET':
        return render_template('models.html')
    

@app.route('/about', methods=['GET'])
def about():
    if request.method=='GET':
        return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)
