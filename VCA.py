import QuantLib as ql
import streamlit as st

######
# Main
######
z='VCA'
st.set_page_config(page_title=z)
st.title(z)

chosenYear = st.radio('Start Year', ['2011', '2016'], index=1)
st.write('')
y = int(chosenYear)
ql.runVCA(y,isSkipTitle=True)