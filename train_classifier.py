import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load extracted features
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100)  # Adjust trees for better performance
model.fit(x_train, y_train)

# Test and print accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model training completed and saved as model.p.")
