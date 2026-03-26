import QuantLib as ql
import streamlit as st

######
# Main
######
z='QS12'
st.set_page_config(page_title=z)
st.title(z)

chosenYear = st.radio('Start Year', ['2008','2016'], index=1)
st.write('')
y = int(chosenYear)
ql.runQS12(y, isSkipTitle=True)
