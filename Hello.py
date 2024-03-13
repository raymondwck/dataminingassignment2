import streamlit as st
from operator import itemgetter
from collections import defaultdict
import pandas as pd
import requests
from io import BytesIO

def recommendFood(user_input, X, features):
    # Now compute for all possible rules
    valid_rules = defaultdict(int)
    invalid_rules = defaultdict(int)
    num_occurences = defaultdict(int)
    n_features = len(features)

    for sample in X:
        for premise in range(n_features):
            if sample[premise] == 0: continue
            # Record that the premise was bought in another transaction
            num_occurences[premise] += 1
            for conclusion in range(n_features):
                if premise == conclusion:  # It makes little sense to measure if X -> X.
                    continue
                if sample[conclusion] == 1:
                    # This person also bought the conclusion item
                    valid_rules[(premise, conclusion)] += 1
                else:
                    # This person bought the premise, but not the conclusion
                    invalid_rules[(premise, conclusion)] += 1
    support = valid_rules
    confidence = defaultdict(float)
    for premise, conclusion in valid_rules.keys():
        confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]

    sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
    sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
    
    def printRule(premise, conclusion, support, confidence, features):
        premise_name = features[premise]
        conclusion_name = features[conclusion]
        rule = f"Rule: If a person buys {premise_name}, they will also buy {conclusion_name}\n"
        rule += f"- Confidence: {confidence[(premise, conclusion)]:.3f}\n"
        rule += f"- Support: {support[(premise, conclusion)]}\n"
        return rule

    # Find the index of the user-input premise in the features list
    premise_index = features.index(user_input)

    # Create a list to store the rules
    rules = []

    # Iterate over the sorted confidence list
    for index in range(len(sorted_confidence)):
        if len(rules) >= 3:
            break

        (premise, conclusion) = sorted_confidence[index][0]
        premise_name = features[premise]
        conclusion_name = features[conclusion]

        # Check if the premise and conclusion names are different and if the premise matches user input
        if premise_name != conclusion_name and premise == premise_index:
            rules.append((premise, conclusion))

    # Sort the rules based on confidence score
    sorted_rules = sorted(rules, key=lambda x: confidence[x], reverse=True)

    # Prepare the rules for display
    rule_texts = []
    for i, rule in enumerate(sorted_rules[:3]):
        rule_texts.append(f"Rule #{i + 1}\n{printRule(rule[0], rule[1], support, confidence, features)}")
        
    return rule_texts

# URL of the Excel file
# url = 'https://github.com/raymondwck/datamining/raw/77e7ff11d72d28afeaa2f850cc03c5bfe6893fc8/JapanMenuItems.xlsx'

# # Download the Excel file from the URL
# response = requests.get(url)
# if response.status_code == 200:
#     # Read the Excel file from the response conte
#     df = pd.read_excel(BytesIO(response.content))
#     # Display the DataFrame
#     print(df)
# else:
#     print("Failed to download the Excel file.")
df = pd.read_excel('JapaneseMenuItems.xlsx')
X = df.values
n_features = 4  # Number of food items
features = ["California Roll", "Salmon Nigiri", "Tonkotsu Ramen", "Chicken Teriyaki Bento", "Edamame", "Gyoza (Dumplings)", "Tempura (Shrimp)", 
            "Green Tea Ice Cream", "Mochi Ice Cream", "Matcha Latte"]

def main():
    st.title("Food Recommendation System")
    # Define your options for the dropdown
    options = {
        "California Roll": 0,
        "Salmon Nigiri": 1,
        "Tonkotsu Ramen": 2,
        "Chicken Teriyaki Bento": 3,
        "Edamame": 4,
        "Gyoza (Dumplings)": 5,
        "Tempura (Shrimp)": 6,
        "Green Tea Ice Cream": 7,
        "Mochi Ice Cream": 8,
        "Matcha Latte": 9
    }
    
        # User input for initial food order using dropdown
    initial_order = st.selectbox("Select your initial food order:", options)
    
    if st.button("Recommend"):
        rules = recommendFood(initial_order, X, features)
        for rule in rules:
            st.write(rule)
            
if __name__ == "__main__":
    main()
