import streamlit as st
from streamlit.logger import get_logger
from diabetesFunction import oneR_predict

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    

    # Define the dictionary for new Apple data
    newApple_data = {
        'Size': [-0.292023862], 
        'Weight': [-1.351281995], 
        'Sweetness': [-1.738429162],  
        'Crunchines': [-0.342615928],  
        'Juiciness': [2.838635512],  
        'Ripeness': [-0.038033328],  
        'Acidity': [-0.038033328]  
    }

    # # Convert the dictionary to a DataFrame
    # new_data = pd.DataFrame(newApple_data)

    # # Use the trained OneR classifier to make predictions
    # new_predictions = oneR_predict(new_data, best_attribute, rules)

    # # Output the predictions
    # st.write("Predictions for the new Apple:")
    # if new_predictions[0] == 0:
    #     st.write("Apple is Good")
    # else:
    #     st.write("Apple is Bad")




if __name__ == "__main__":
    run()
