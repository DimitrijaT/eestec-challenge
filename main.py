import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pydeck as pdk
import numpy as np
import plotly.express as px
import cv2
from prophet.plot import plot_plotly, plot_components_plotly

sensor_id_locations = {
    "Lisice": "f9a91b1f-a6cf-4484-90e4-e5df7a128625",
    "Petrovec": "756cc065-dc00-4558-9195-582c527de537",
    "Dracevo": "b79604bb-0bea-454f-a474-849156e418ea",
    "Kisela_Voda": "sensor_dev_77790_464",
    # "Centar":"0a058579-12c9-47be-971b-607198002d3b",
    "Karposh": "cec29ba1-5414-4cf3-bbcc-8ce4db1da5d0",
    "Gjorce": "768284ed-72be-4c18-b764-1f9de38b365f",
    "Butel": "3d7bd712-24a9-482c-b387-a8168b12d3f4",
    "Radishani": "07b58ccf-7faa-4f0a-a10a-e7b485d52ffe",
}

sensor_info = pd.read_csv("content/sensor_info.csv")


def get_location(sensor_arg):
    sensor_id_arg = sensor_id_locations[sensor_arg]
    position = sensor_info[sensor_info['SensorId'] == sensor_id_arg]['Position'].values[0]
    latitude, longitude = position.split(',')
    latitude = float(latitude)
    longitude = float(longitude)
    return {"Latitude": latitude,
            "Longitude": longitude}


def visualize_data(data_arg):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
    data_arg.groupby('Type')['Value'].plot(legend=True)
    plt.title('Sensor Data')
    plt.xlabel('Stamp')
    plt.ylabel('Value')
    plt.grid(True)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    return fig


def calculate_accuracy(forecast_df):
    actual_values = data['Value']
    predicted_values = forecast_df['yhat'][:-int(daysToPredict)]
    # st.write(actual_values)
    # st.write(predicted_values)
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    mae = np.mean(np.abs(actual_values - predicted_values))
    return mape, mae


def preprocess_data(data_r, type_s, from_date_arg, to_date_arg):
    df = data_r.copy()
    type_s = type_s[2:].lower()
    df = df[df["Type"] == type_s]
    df['Stamp'] = pd.to_datetime(df['Stamp'], utc=True).dt.tz_convert(None)
    df = df[(df['Stamp'] >= from_date_arg) & (df['Stamp'] <= to_date_arg)]
    df = df.drop(['SensorId', 'Type'], axis=1)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.columns = ['y', 'ds']
    df.reset_index(drop=True, inplace=True)
    return df


c1, c2, c3, c4, c5 = st.columns(5)
with c3:
    st.image('content/logo2.png', width=180)

st.markdown("<h1 style='text-align: center; color: grey;'>PulseEco Forecaster</h1>", unsafe_allow_html=True)

st.subheader("Пополни ја формата за предвидување на избраната категорија")

option = st.selectbox(
    'Select municipality',
    ("Lisice",
     "Petrovec",
     "Dracevo",
     "Kisela_Voda",
     "Karposh",
     "Gjorce",
     "Butel",
     "Radishani",))

st.write('You selected:', option)

viz_fig = None

if option is not None:
    data_path = "content/Datasets/" + option + ".csv"
    data = pd.read_csv(data_path)

    # Create an empty DataFrame to store latitude and longitude information
    sensor_location_df = pd.DataFrame(columns=["Sensor", "Latitude", "Longitude", "size", "color"])

    # Iterate over sensor_id_locations dictionary and populate the DataFrame
    for sensor, sensor_id in sensor_id_locations.items():
        location = get_location(sensor)
        if sensor == option:
            size = 1000
            color = "#f45759"
        else:
            size = 600
            color = "#17163f"
        new_row = pd.DataFrame({"Sensor": [sensor],
                                "latitude": [location["Latitude"]],
                                "longitude": [location["Longitude"]],
                                "size": [size],
                                "color": [color]})
        sensor_location_df = pd.concat([sensor_location_df, new_row], ignore_index=True)

    st.map(sensor_location_df, size="size", color="color", use_container_width=True)

    if st.button("Visualize"):
        viz_fig = visualize_data(data)
    if viz_fig is not None:
        st.pyplot(viz_fig)

    first_date = pd.to_datetime(data['Stamp'][0])
    last_date = pd.to_datetime(data.tail(1)['Stamp'].values[0])

    from_date = st.date_input("From Date", min_value=first_date, value=first_date, max_value=last_date)
    to_date = st.date_input("To Date", max_value=last_date, value=last_date, min_value=from_date)

    daysToPredict = st.text_input("How many days to predict in the future?", value=30)

    typeS = st.radio(
        'Select a parameter',
        ['💧 Humidity', '🦠 pm10', '😷 pm25', '🌡️ Temperature']
    )

    toShow = typeS[2:]

    st.write("You selected: :rainbow[" + toShow + "]")

    # Button to trigger data preprocessing and training
    if st.button('Train'):
        processed_data = preprocess_data(data, typeS, pd.to_datetime(from_date), pd.to_datetime(to_date))
    else:
        processed_data = None


def plot_forecast(processed_data):
    if processed_data is not None:
        with st.spinner('Wait for it...'):
            m = Prophet()
            # processed_data['cap'] = 500
            if typeS != 'temperature':
                processed_data['floor'] = 0.0
            m.fit(processed_data)
            future = m.make_future_dataframe(periods=int(daysToPredict))
            # future['cap'] = 1000
            if typeS != 'temperature':
                future['floor'] = 0.0
            forecast = m.predict(future)

            fig1 = m.plot(forecast)

            fig2 = m.plot_components(forecast)

            from prophet.plot import plot_plotly, plot_components_plotly

            fig_plotly = px.line(forecast, x='ds', y='yhat', title='Prophet Forecast')
            fig_plotly.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
            fig_plotly.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound')
            fig_plotly.update_xaxes(title='Date')
            fig_plotly.update_yaxes(title='Value')

    return fig1, fig2, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fig_plotly


def main():
    if processed_data is not None:
        fig1, fig2, forecast_df, fig_plotly = plot_forecast(processed_data)
        st.balloons()

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Components", "📄Forecast Dataframe", "🎯Accuracy"])

        tab1.subheader("Prophet chart")
        tab1.pyplot(fig1)
        tab1.subheader("Plotly Chart")
        tab1.plotly_chart(fig_plotly)

        tab2.subheader("Components charts")
        tab2.pyplot(fig2)

        tab3.subheader("Forecast Dataframe")
        tab3.write(forecast_df)

        tab4.subheader("Accuracy Metrics")
        mape, mae = calculate_accuracy(forecast_df)
        tab4.write("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))
        tab4.write("Mean Absolute Error (MAE): {:.2f}".format(mae))

    else:
        st.write("Click 'Train' to preprocess and train the data.")


if __name__ == '__main__':
    main()
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    c1, c2, c3 = st.columns(3)
    with c2:
        st.markdown("<h5 style='text-align: center; color: grey;'>EESTEC 2024 Hackathon</h5>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: grey;'>Made with 💖 by Dimitrija, Andrej & Filip</h6>",
                    unsafe_allow_html=True)
