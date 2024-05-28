import streamlit as st
import pandas as pd
df = pd.DataFrame({
 'first column': [1, 2, 3, 4],
 'second column': [10, 20, 30, 40]
})
st.write("Here's our first table:")
st.write(df)

import streamlit as st
import numpy as np
import pandas as pd
chart_data = pd.DataFrame(
 np.random.randn(20, 3),
 columns=['a', 'b', 'c'])
st.line_chart(chart_data)

map_data = pd.DataFrame(
 np.random.randn(1000, 2) / [50, 50] + [13.76514, 100.53829],
 columns=['lat', 'lon'])
st.map(map_data)