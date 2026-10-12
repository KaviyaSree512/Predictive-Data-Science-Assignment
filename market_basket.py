
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample market basket transactions
data = {
    "Milk":  [1, 1, 1, 0, 1],
    "Bread": [1, 1, 0, 1, 1],
    "Butter":[0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Find frequent itemsets
frequent_items = apriori(
    df.astype(bool),
    min_support=0.4,
    use_colnames=True
)

print("Market Basket Analysis")
print("----------------------")
print("\nFrequent Itemsets:")
print(frequent_items)

# Generate association rules
rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules:")
print(rules[["antecedents", "consequents",
             "support", "confidence"]])
