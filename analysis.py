import pandas as pd
import matplotlib.pyplot as plt

# Create dataset
data = {
    "Name": ["Ali", "Sara", "Yousef", "Mona"],
    "Math": [80, 90, 85, 70],
    "Science": [75, 95, 80, 65]
}

df = pd.DataFrame(data)

# Show data
print("Dataset:")
print(df)

# Analysis
print("\nAverage Math Score:", df["Math"].mean())
print("Average Science Score:", df["Science"].mean())

# Best student in Math
top_student = df[df["Math"] == df["Math"].max()]
print("\nTop Math Student:")
print(top_student)

# Visualization
plt.bar(df["Name"], df["Math"])
plt.title("Math Scores")
plt.xlabel("Students")
plt.ylabel("Score")
plt.show()
