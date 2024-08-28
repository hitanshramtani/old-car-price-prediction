import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# import sklearn
df = pd.read_csv("carpredictor3madebymecleaned.csv")

df = df.iloc[:,1:]
# df

model = pickle.load(open('CarSalePriceXGB.pkl','rb'))



st.title("Car Price Predictor")
st.write("""It is a Old car price detector where you have to provide some details and this
          model will tell you the price of the given input.""")
""
st.subheader("Lets Predict")
cp = st.selectbox("**Company**",df["company"].unique(),index = None)
if cp:
    st.write(cp)
""
if cp:
    filtered_models = df[df["car_name"].str.startswith(cp)]["car_name"].unique()
else:
    filtered_models = df["car_name"].unique()
mn = st.selectbox("**Model**", filtered_models, index=None)
if mn:
    st.write(mn)
""
yrdriven =st.slider("**Car Operating Years**",min_value = 0,max_value=60,value = 0,step = 1)
manufactureyear = 2024-yrdriven
st.write("**Manufacture Year**",manufactureyear)
""
ft = st.radio("**Fuel Type**",["Petrol","Diesel"],index = None)
if ft:
    st.write(ft)
""
km= st.text_input("**Total KMs Driven**")
st.write(km,"Kms")
""

if st.button("**Price Predictor**"):
    input_data = pd.DataFrame({
        'car_name': [mn],
        'company': [cp],
        'year': [manufactureyear],
        'kms_driven': [km],
        'fuel_type': [ft]
    })
    prediction = model.predict(input_data)

    st.write(f'Estimated Car Price: {prediction[0]:,.2f}rs')


st.markdown("""
    <br><br><br><br><br><br>
""", unsafe_allow_html=True)
chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('year:O'),  # Use 'O' for ordinal if treating 'year' as categorical
    y=alt.Y('Price:Q'), # Use 'Q' for quantitative
    color='company:N',  # Nominal scale for 'company'
    tooltip=['year', 'Price', 'company']  # Tooltip with these fields
)

st.altair_chart(chart, use_container_width=True)
chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('kms_driven'),  # Use 'O' for ordinal if treating 'year' as categorical
    y=alt.Y('Price:Q'), # Use 'Q' for quantitative
    color='company:N',  # Nominal scale for 'company'
    tooltip=['year', 'Price', 'company']  # Tooltip with these fields
)

st.altair_chart(chart, use_container_width=True)

fig = plt.figure(figsize=(14, 7))
ax = sns.relplot(x='company', y='Price', data=df, hue='fuel_type', size='year', height=7, aspect=2)
ax.set_xticklabels(rotation=40, ha='right')
plt.tight_layout()
st.pyplot(ax)

col1, col2 = st.columns([1,3])

# First Graphviz chart in the first column
with col1:
    st.graphviz_chart("""
    digraph {
        Company -> Model_Name;
        Model_Name -> Manufacture_Year;
        Manufacture_Year -> Fuel;
        Fuel -> Total_KMs_Driven;
        Total_KMs_Driven -> Price;
    }
    """,use_container_width = True)

# Second Graphviz chart in the second column
with col2:
    st.graphviz_chart("""
    digraph {
        Cleaning_of_data -> Handling_Mising_value,  "Filling_NaN_Values",Removing_Bad_Values  ;
    }""",use_container_width = True)

